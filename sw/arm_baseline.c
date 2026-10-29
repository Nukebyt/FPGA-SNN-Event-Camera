/* arm_baseline.c -- the whole pipeline in software, for the "FPGA versus ARM" comparison (roadmap 7.3).
 *
 * Same function as the FPGA engine, bit for bit: frame (128x128 luma) -> log LUT -> per-pixel reference compare ("step" mode) ->
 * 2x32x32 presence bins -> two LIF conv layers with OR pooling -> classifier -> window of 32 frames. Integer arithmetic only,
 * the same network weights (exported by scripts/export_arm_baseline.py), results checked against model/snn_int.py.
 *
 * Two implementations of the network so the comparison is fair:
 *   --mode dense   output-stationary, the algorithm the FPGA runs (every neuron sums all its inputs; plain loops)
 *   --mode sparse  input-stationary / event driven, what one would write for a CPU: each active input bit adds its weights
 *                  into the accumulators of the neurons it touches (channels last so the inner loop is contiguous)
 *
 * Build:   gcc -std=gnu99 -O3 -Wall -Wextra -o arm_baseline arm_baseline.c        (on the board: add -mcpu=cortex-a9 -mfpu=neon)
 * Run:     ./arm_baseline --weights w.bin --lut lut.bin --frames f.u8 --n 300 --mode sparse --out res.bin
 * Files:   w.bin = int8 w1[16][2][3][3], w2[32][16][3][3], fc[6][2048] (numpy order); lut.bin = uint8[256]; f.u8 = N x 16384 bytes
 * Output:  res.bin = per frame 12 int32: spk1, spk2, events, cur[6], pred, windows, wcount (same fields as the board's RESULT)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define C1 16
#define C2 32
#define NCLS 6
#define T_WIN 32
#define NPIX 16384
#define FC_IN 2048
#define THR 64
#define VMIN (-2048)
#define VMAX 2047

static int8_t w1[C1][2][3][3], w2[C2][C1][3][3], wfc[NCLS][FC_IN];
static int8_t w1c[2 * 9][C1];            /* [ci*9 + ky*3 + kx][o] channels last, for the sparse path */
static int8_t w2c[C1 * 9][C2];
static int8_t wfcc[FC_IN][NCLS];
static uint8_t lut[256];

static int16_t ref[NPIX];
static int have_ref = 0;
static int theta = 9;

static uint8_t bins[2][32][32];
static int16_t v1[32 * 32][C1];          /* [y*32+x][channel] */
static int16_t v2[16 * 16][C2];
static uint8_t pool1[C1][16][16];
static uint8_t pool2[C2][8][8];
static int16_t acc1[32 * 32][C1];
static int16_t acc2[16 * 16][C2];
static int64_t score[NCLS];
static int wcount = 0, windows = 0, pred = 0;

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int emulate(const uint8_t *frame)
{
    int events = 0;
    memset(bins, 0, sizeof(bins));
    if (!have_ref) {
        for (int i = 0; i < NPIX; i++) ref[i] = lut[frame[i]];
        have_ref = 1;
        return 0;
    }
    for (int y = 0; y < 128; y++)
        for (int x = 0; x < 128; x++) {
            int i = y * 128 + x, cur = lut[frame[i]], d = cur - ref[i];
            if (d >= theta) { ref[i] += theta; bins[1][y >> 2][x >> 2] = 1; events++; }
            else if (d <= -theta) { ref[i] -= theta; bins[0][y >> 2][x >> 2] = 1; events++; }
        }
    return events;
}

static inline int lif(int16_t *v, int acc, int *spike)
{
    int t = *v;
    t -= t >> 4;
    t += acc;
    if (t < VMIN) t = VMIN;
    if (t > VMAX) t = VMAX;
    *spike = t >= THR;
    if (*spike) t -= THR;
    *v = (int16_t)t;
    return *spike;
}

/* ---------------------------------------------------------------- sparse (event driven) */
static void step_sparse(int *spk1, int *spk2, int32_t cur[NCLS])
{
    memset(acc1, 0, sizeof(acc1));
    for (int ci = 0; ci < 2; ci++)
        for (int yi = 0; yi < 32; yi++)
            for (int xi = 0; xi < 32; xi++) {
                if (!bins[ci][yi][xi]) continue;
                for (int ky = 0; ky < 3; ky++) {
                    int y = yi - ky + 1;
                    if (y < 0 || y >= 32) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int x = xi - kx + 1;
                        if (x < 0 || x >= 32) continue;
                        int16_t *a = acc1[y * 32 + x];
                        const int8_t *w = w1c[ci * 9 + ky * 3 + kx];
                        for (int o = 0; o < C1; o++) a[o] += w[o];
                    }
                }
            }
    memset(pool1, 0, sizeof(pool1));
    int s1 = 0;
    for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++)
            for (int o = 0; o < C1; o++) {
                int sp;
                lif(&v1[y * 32 + x][o], acc1[y * 32 + x][o], &sp);
                s1 += sp;
                pool1[o][y >> 1][x >> 1] |= (uint8_t)sp;
            }
    *spk1 = s1;

    memset(acc2, 0, sizeof(acc2));
    for (int ci = 0; ci < C1; ci++)
        for (int yi = 0; yi < 16; yi++)
            for (int xi = 0; xi < 16; xi++) {
                if (!pool1[ci][yi][xi]) continue;
                for (int ky = 0; ky < 3; ky++) {
                    int y = yi - ky + 1;
                    if (y < 0 || y >= 16) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int x = xi - kx + 1;
                        if (x < 0 || x >= 16) continue;
                        int16_t *a = acc2[y * 16 + x];
                        const int8_t *w = w2c[ci * 9 + ky * 3 + kx];
                        for (int o = 0; o < C2; o++) a[o] += w[o];
                    }
                }
            }
    memset(pool2, 0, sizeof(pool2));
    int s2 = 0;
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++)
            for (int o = 0; o < C2; o++) {
                int sp;
                lif(&v2[y * 16 + x][o], acc2[y * 16 + x][o], &sp);
                s2 += sp;
                pool2[o][y >> 1][x >> 1] |= (uint8_t)sp;
            }
    *spk2 = s2;

    for (int k = 0; k < NCLS; k++) cur[k] = 0;
    for (int c = 0; c < C2; c++)
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                if (pool2[c][y][x]) {
                    const int8_t *w = wfcc[c * 64 + y * 8 + x];
                    for (int k = 0; k < NCLS; k++) cur[k] += w[k];
                }
}

/* ---------------------------------------------------------------- dense (what the FPGA does) */
static void step_dense(int *spk1, int *spk2, int32_t cur[NCLS])
{
    static uint8_t in1[2][34][34], in2[C1][18][18];
    memset(in1, 0, sizeof(in1));
    for (int ci = 0; ci < 2; ci++)
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++) in1[ci][y + 1][x + 1] = bins[ci][y][x];
    memset(pool1, 0, sizeof(pool1));
    int s1 = 0;
    for (int o = 0; o < C1; o++)
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++) {
                int acc = 0;
                for (int ci = 0; ci < 2; ci++)
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++) acc += in1[ci][y + ky][x + kx] * w1[o][ci][ky][kx];
                int sp;
                lif(&v1[y * 32 + x][o], acc, &sp);
                s1 += sp;
                pool1[o][y >> 1][x >> 1] |= (uint8_t)sp;
            }
    *spk1 = s1;
    memset(in2, 0, sizeof(in2));
    for (int ci = 0; ci < C1; ci++)
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++) in2[ci][y + 1][x + 1] = pool1[ci][y][x];
    memset(pool2, 0, sizeof(pool2));
    int s2 = 0;
    for (int o = 0; o < C2; o++)
        for (int y = 0; y < 16; y++)
            for (int x = 0; x < 16; x++) {
                int acc = 0;
                for (int ci = 0; ci < C1; ci++)
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++) acc += in2[ci][y + ky][x + kx] * w2[o][ci][ky][kx];
                int sp;
                lif(&v2[y * 16 + x][o], acc, &sp);
                s2 += sp;
                pool2[o][y >> 1][x >> 1] |= (uint8_t)sp;
            }
    *spk2 = s2;
    for (int k = 0; k < NCLS; k++) {
        int a = 0;
        for (int i = 0; i < FC_IN; i++) a += ((const uint8_t *)pool2)[i] * wfc[k][i];
        cur[k] = a;
    }
}

static int load(const char *path, void *dst, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    size_t got = fread(dst, 1, n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "%s: %zu bytes, expected %zu\n", path, got, n); return -1; }
    return 0;
}

static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv)
{
    const char *wf = NULL, *lf = NULL, *ff = NULL, *of = NULL, *mode = "sparse";
    int n = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--weights") && i + 1 < argc) wf = argv[++i];
        else if (!strcmp(argv[i], "--lut") && i + 1 < argc) lf = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) ff = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) of = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --weights w.bin --lut lut.bin --frames f.u8 --n N [--mode dense|sparse] [--out res.bin]\n", argv[0]); return 2; }
    }
    if (!wf || !lf || !ff || n <= 0) { fprintf(stderr, "missing arguments\n"); return 2; }
    int dense = !strcmp(mode, "dense");
    if (!dense && strcmp(mode, "sparse")) { fprintf(stderr, "mode must be dense or sparse\n"); return 2; }

    uint8_t *wb = (uint8_t *)malloc(288 + 4608 + 12288);
    if (!wb || load(wf, wb, 288 + 4608 + 12288) || load(lf, lut, 256)) return 1;
    memcpy(w1, wb, 288);
    memcpy(w2, wb + 288, 4608);
    memcpy(wfc, wb + 288 + 4608, 12288);
    free(wb);
    for (int ci = 0; ci < 2; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) for (int o = 0; o < C1; o++) w1c[ci * 9 + ky * 3 + kx][o] = w1[o][ci][ky][kx];
    for (int ci = 0; ci < C1; ci++) for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) for (int o = 0; o < C2; o++) w2c[ci * 9 + ky * 3 + kx][o] = w2[o][ci][ky][kx];
    for (int i = 0; i < FC_IN; i++) for (int k = 0; k < NCLS; k++) wfcc[i][k] = wfc[k][i];

    uint8_t *frames = (uint8_t *)malloc((size_t)n * NPIX);
    int32_t *res = (int32_t *)malloc((size_t)n * 12 * sizeof(int32_t));
    double *te = (double *)malloc(n * sizeof(double)), *ts = (double *)malloc(n * sizeof(double)), *tt = (double *)malloc(n * sizeof(double));
    if (!frames || !res || !te || !ts || !tt || load(ff, frames, (size_t)n * NPIX)) return 1;

    for (int f = 0; f < n; f++) {
        int spk1 = 0, spk2 = 0;
        int32_t cur[NCLS];
        double t0 = now_us();
        int events = emulate(frames + (size_t)f * NPIX);
        double t1 = now_us();
        if (dense) step_dense(&spk1, &spk2, cur); else step_sparse(&spk1, &spk2, cur);
        double t2 = now_us();
        for (int k = 0; k < NCLS; k++) score[k] += cur[k];
        wcount++;
        if (wcount == T_WIN) {
            int best = 0;
            for (int k = 1; k < NCLS; k++) if (score[k] > score[best]) best = k;
            pred = best;
            windows++;
            wcount = 0;
            memset(score, 0, sizeof(score));
            memset(v1, 0, sizeof(v1));
            memset(v2, 0, sizeof(v2));
        }
        int32_t *r = res + (size_t)f * 12;
        r[0] = spk1; r[1] = spk2; r[2] = events;
        for (int k = 0; k < NCLS; k++) r[3 + k] = cur[k];
        r[9] = pred; r[10] = windows; r[11] = wcount;
        te[f] = t1 - t0; ts[f] = t2 - t1; tt[f] = t2 - t0;
    }
    if (of) {
        FILE *fo = fopen(of, "wb");
        if (!fo) { perror(of); return 1; }
        fwrite(res, sizeof(int32_t), (size_t)n * 12, fo);
        fclose(fo);
    }
    double sum_t = 0, sum_e = 0, sum_s = 0;
    for (int f = 0; f < n; f++) { sum_t += tt[f]; sum_e += te[f]; sum_s += ts[f]; }
    qsort(tt, n, sizeof(double), cmp_d);
    printf("mode %s, %d frames: per frame emulator %.0f us + network %.0f us = %.0f us (median %.0f, p99 %.0f, max %.0f) -> %.1f frames/s\n",
           mode, n, sum_e / n, sum_s / n, sum_t / n, tt[n / 2], tt[(int)(n * 0.99)], tt[n - 1], 1e6 * n / sum_t);
    return 0;
}
