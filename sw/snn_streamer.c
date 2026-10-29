/* snn_streamer.c -- frame streaming daemon for the event-camera SNN engine, run on the DE10-Standard HPS (roadmap Phase 1 / 6).
 *
 * Derived from the night-vision project's nv_streamer.c, which ran on this board (same structure: the HPS image has no usable
 * Python, so this C daemon owns /dev/mem and the laptop talks to it over TCP, docs/snn_soc.md). One connection at a time, SIGPIPE
 * ignored, TCP_NODELAY, magic/version header, error codes. Build with -std=gnu99 (the board's gcc defaults to gnu89). Root needed.
 *
 *   laptop --FRAME_IN (128x128 luma)--> daemon --memcpy--> DDR3 input ring --FPGA dma_rd--> snn_engine
 *   laptop <--RESULT (class, scores, counters)-- daemon <-- CSR block (snapshot of the finished frame)
 *
 * Memory: the input ring (4 slots x 16 KiB) lives in DDR3 that Linux must not use: on this board the device tree limits Linux to
 * 768 MB, so 0x30000000 is free (docs/board_notes.md of the night-vision project). Uncached /dev/mem mapping (O_SYNC) because the
 * FPGA reads DDR3 behind the CPU caches.
 *
 * Build:   gcc -std=gnu99 -O2 -Wall -Wextra -o snn_streamer snn_streamer.c            (on the board; it has gcc)
 * Test without hardware (fake engine, PC):   gcc -std=gnu99 -O2 -DSNN_SIM -o snn_streamer_sim snn_streamer.c
 * Run:     ./snn_streamer [--port 5002] [--ddr-base 0x30000000]
 */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

/* ---------------------------------------------------------------- hardware map (keep in sync with rtl/snn_engine.v) */
#define HPS_LW_BRIDGE_BASE 0xFF200000UL
#define HPS_LW_BRIDGE_SPAN 0x00200000UL
#define SNN_CSR_OFFSET     0x00010000UL          /* snn_engine_0.csr inside the lightweight bridge (scripts/make_snn_soc.py) */
#define SNN_EXPECTED_BUILD 0x534E0001u
#define SLOT_STRIDE        0x4000UL              /* 128*128 bytes */
#define N_SLOTS            4
#define DDR_REGION_SPAN    0x10000UL

#define R_VERSION   0
#define R_CONTROL   1
#define R_STATUS    2
#define R_STATUS_CLR 3
#define R_IN_BASE   4
#define R_IN_WR_SEQ 6
#define R_IN_RD_SEQ 7
#define R_DONE_SEQ  8
#define R_THETA     9
#define R_SCRATCH   11
#define R_WINDOW_T  12
#define R_STEP_CYC  13
#define R_LAT_CYC   14
#define R_PRED      16
#define R_WINDOWS   17
#define R_WCOUNT    18
#define R_SPK1      19
#define R_SPK2      20
#define R_EVENTS    21
#define R_WIN_SCORE 24
#define R_CUR       32
#define R_SCORE     40
#define NCLS        6

#define CTL_ENABLE  (1u << 0)
#define CTL_RESET   (1u << 1)
#define CTL_CLRREF  (1u << 2)
#define CTL_MODE    (1u << 3)
#define ST_READY    (1u << 0)

/* ---------------------------------------------------------------- protocol (docs/snn_soc.md; same framing as the night-vision one) */
#define SN_MAGIC   0x454E4E53u            /* 'S','N','N','E' little-endian */
#define SN_VERSION 1u
#define T_HELLO 1
#define T_HELLO_ACK 2
#define T_FRAME_IN 3
#define T_RESULT 4
#define T_CSR_WRITE 5
#define T_CSR_READ 6
#define T_CSR_REPLY 7
#define T_ERROR 8
#define T_RESET 9                         /* laptop -> board: restart the engine (new clip), answered with HELLO_ACK */
#define E_BAD_MAGIC 1u
#define E_BAD_VERSION 2u
#define E_BAD_SIZE 3u
#define E_SHORT_READ 4u
#define E_NOMEM 5u
#define E_BUSY 6u
#define E_FPGA_TIMEOUT 7u
#define FRAME_W 128
#define FRAME_H 128
#define NPIX (FRAME_W * FRAME_H)
#define RESULT_WORDS 28                   /* 10 scalars + 3 x 6 class values, see send_result() */

static volatile uint32_t *csr;            /* CSR block, 32-bit words */
static volatile uint8_t  *ddr;            /* mapped reserved DDR3 region */
static uint32_t ddr_phys = 0x30000000u;

#ifdef SNN_SIM
/* ---------------------------------------------------------------- fake engine so the daemon and the protocol can be tested on a PC.
 * It does NOT run the network: after every queued frame it publishes class = frame count mod 6 and counters derived from the
 * first pixel, enough for tests/test_snn_streamer.py to check framing, sequencing and register plumbing. */
static uint32_t sim_regs[256];
static void sim_on_write(unsigned w, uint32_t v)
{
    if (w == R_IN_WR_SEQ) {
        while (sim_regs[R_IN_RD_SEQ] < v) {
            unsigned s = sim_regs[R_IN_RD_SEQ] % N_SLOTS;
            uint32_t first = ddr[s * SLOT_STRIDE];
            sim_regs[R_IN_RD_SEQ]++;
            sim_regs[R_DONE_SEQ]++;
            sim_regs[R_PRED] = sim_regs[R_DONE_SEQ] % NCLS;
            sim_regs[R_WINDOWS] = sim_regs[R_DONE_SEQ] / 32;
            sim_regs[R_WCOUNT] = sim_regs[R_DONE_SEQ] % 32;
            sim_regs[R_SPK1] = first; sim_regs[R_SPK2] = first / 2; sim_regs[R_EVENTS] = first * 3;
            sim_regs[R_STEP_CYC] = 205832; sim_regs[R_LAT_CYC] = 222233;
            for (int k = 0; k < NCLS; k++) {
                sim_regs[R_CUR + k] = (uint32_t)(int32_t)(100 * (k + 1) - (int)first);
                sim_regs[R_SCORE + k] = (uint32_t)(int32_t)(-1000 * (k + 1));
                sim_regs[R_WIN_SCORE + k] = (uint32_t)(int32_t)(7 * k);
            }
        }
    } else if (w == R_CONTROL && (v & CTL_RESET)) {
        sim_regs[R_IN_RD_SEQ] = sim_regs[R_DONE_SEQ] = sim_regs[R_IN_WR_SEQ] = 0;
        sim_regs[R_STATUS] = ST_READY;
    }
}
static inline uint32_t csr_rd(unsigned w) { (void)csr; return sim_regs[w]; }
static inline void csr_wr(unsigned w, uint32_t v)
{
    if (w != R_IN_RD_SEQ && w != R_DONE_SEQ && w != R_VERSION && w != R_STATUS) sim_regs[w] = v;
    sim_on_write(w, v);
}
#else
static inline uint32_t csr_rd(unsigned w) { return csr[w]; }
static inline void csr_wr(unsigned w, uint32_t v) { csr[w] = v; }
#endif

/* ---------------------------------------------------------------- io helpers */
static int read_exact(int fd, void *buf, size_t n)
{
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r == 0) return -1;
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        got += (size_t)r;
    }
    return 0;
}
static int write_exact(int fd, const void *buf, size_t n)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, p + sent, n - sent);
        if (w <= 0) { if (w < 0 && errno == EINTR) continue; return -1; }
        sent += (size_t)w;
    }
    return 0;
}
static long now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

static void put16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { put16(p, v); put16(p + 2, v >> 16); }
static uint32_t get32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint32_t get16(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }

static int send_msg(int fd, uint16_t type, uint32_t seq, const void *payload, uint32_t len)
{
    /* header and payload go out in ONE write: two small writes made a 40 ms delay per message (delayed ACK against the laptop,
     * measured on the board, docs/bug_log.md #31) */
    uint8_t m[16 + 256];
    put32(m, SN_MAGIC); put16(m + 4, SN_VERSION); put16(m + 6, type); put32(m + 8, seq); put32(m + 12, len);
    if (len > 256) return -1;
    if (len) memcpy(m + 16, payload, len);
    return write_exact(fd, m, 16 + (size_t)len);
}
static void send_error(int fd, uint32_t code)
{
    uint8_t p[4];
    put32(p, code);
    (void)send_msg(fd, T_ERROR, 0, p, 4);
}

/* ---------------------------------------------------------------- DDR3 ring access (64-bit stores, uncached mapping) */
static void ddr_write(unsigned slot, const uint8_t *src)
{
    volatile uint64_t *d = (volatile uint64_t *)(ddr + (size_t)slot * SLOT_STRIDE);
    for (size_t i = 0; i < NPIX / 8; i++) {
        uint64_t v;
        memcpy(&v, src + 8 * i, 8);                 /* source buffer is not guaranteed 8-byte aligned */
        d[i] = v;
    }
}

/* ---------------------------------------------------------------- engine control */
static int engine_start(void)
{
    csr_wr(R_CONTROL, CTL_RESET);                   /* soft reset, engine disabled */
    long t0 = now_ms();
    usleep(2000);
    while (!(csr_rd(R_STATUS) & ST_READY)) {        /* ready = builder swept, core cleared (about 0.3 ms) */
        if (now_ms() - t0 > 2000) return -1;
        usleep(1000);
    }
    csr_wr(R_STATUS_CLR, 1);
    csr_wr(R_IN_BASE, ddr_phys);
    csr_wr(R_CONTROL, CTL_ENABLE);
    return 0;
}
static void engine_stop(void) { csr_wr(R_CONTROL, 0); }

/* RESULT payload, 28 little-endian 32-bit words:
 *  0 done_seq  1 pred  2 windows  3 wcount  4 spk1  5 spk2  6 events  7 step_cycles  8 latency_cycles  9 status
 *  10..15 current of each class for this frame     16..21 running window score     22..27 score of the last finished window */
static int send_result(int fd, uint32_t seq)
{
    uint8_t r[RESULT_WORDS * 4];
    uint32_t v[RESULT_WORDS];
    v[0] = csr_rd(R_DONE_SEQ); v[1] = csr_rd(R_PRED); v[2] = csr_rd(R_WINDOWS); v[3] = csr_rd(R_WCOUNT);
    v[4] = csr_rd(R_SPK1); v[5] = csr_rd(R_SPK2); v[6] = csr_rd(R_EVENTS); v[7] = csr_rd(R_STEP_CYC);
    v[8] = csr_rd(R_LAT_CYC); v[9] = csr_rd(R_STATUS);
    for (int k = 0; k < NCLS; k++) {
        v[10 + k] = csr_rd(R_CUR + (unsigned)k);
        v[16 + k] = csr_rd(R_SCORE + (unsigned)k);
        v[22 + k] = csr_rd(R_WIN_SCORE + (unsigned)k);
    }
    for (int i = 0; i < RESULT_WORDS; i++) put32(r + 4 * i, v[i]);
    return send_msg(fd, T_RESULT, seq, r, sizeof(r));
}

/* ---------------------------------------------------------------- one connection */
static void handle_conn(int fd)
{
    uint8_t h[16];
    if (read_exact(fd, h, 16) != 0) return;
    if (get32(h) != SN_MAGIC) { send_error(fd, E_BAD_MAGIC); return; }
    if (get16(h + 4) != SN_VERSION) { send_error(fd, E_BAD_VERSION); return; }
    uint32_t type = get16(h + 6), len = get32(h + 12);
    if (type != T_HELLO || len != 8) { send_error(fd, E_BAD_SIZE); return; }
    uint8_t hp[8];
    if (read_exact(fd, hp, 8) != 0) return;
    uint32_t w = get16(hp), hh = get16(hp + 2);
    if (w != FRAME_W || hh != FRAME_H) { send_error(fd, E_BAD_SIZE); return; }

    uint32_t build = csr_rd(R_VERSION);
    printf("  engine VERSION 0x%08X (expected 0x%08X)\n", build, SNN_EXPECTED_BUILD);
    if (build != SNN_EXPECTED_BUILD) fprintf(stderr, "  WARNING: bitstream build id does not match this daemon\n");
    if (engine_start() != 0) { send_error(fd, E_FPGA_TIMEOUT); return; }

    uint8_t ack[12];
    put32(ack, 0); put32(ack + 4, build); put16(ack + 8, FRAME_W); put16(ack + 10, FRAME_H);
    if (send_msg(fd, T_HELLO_ACK, 0, ack, 12) != 0) return;

    uint8_t *inbuf = (uint8_t *)malloc(NPIX + 16);
    if (!inbuf) { send_error(fd, E_NOMEM); return; }
    uint32_t seq_ring[N_SLOTS] = {0};
    uint32_t in_wr = 0, out_done = 0;

    for (;;) {
        struct pollfd pf = { fd, POLLIN, 0 };
        int pr = poll(&pf, 1, 1);
        if (pr < 0 && errno != EINTR) break;
        if (pr > 0) {
            if (pf.revents & (POLLHUP | POLLERR)) break;
            if (read_exact(fd, h, 16) != 0) break;
            if (get32(h) != SN_MAGIC) { send_error(fd, E_BAD_MAGIC); break; }
            if (get16(h + 4) != SN_VERSION) { send_error(fd, E_BAD_VERSION); break; }
            type = get16(h + 6); uint32_t seq = get32(h + 8); len = get32(h + 12);
            if (type == T_FRAME_IN) {
                if (len != NPIX) { send_error(fd, E_BAD_SIZE); break; }
                if (read_exact(fd, inbuf, NPIX) != 0) break;
                if (in_wr - csr_rd(R_IN_RD_SEQ) >= N_SLOTS - 1) { send_error(fd, E_BUSY); break; }
                unsigned slot = in_wr % N_SLOTS;
                ddr_write(slot, inbuf);
                seq_ring[slot] = seq;
                __sync_synchronize();
                in_wr++;
                csr_wr(R_IN_WR_SEQ, in_wr);
            } else if (type == T_RESET && len == 0) {
                if (engine_start() != 0) { send_error(fd, E_FPGA_TIMEOUT); break; }
                in_wr = 0; out_done = 0;
                memset(seq_ring, 0, sizeof(seq_ring));
                put32(ack, 0); put32(ack + 4, build); put16(ack + 8, FRAME_W); put16(ack + 10, FRAME_H);
                if (send_msg(fd, T_HELLO_ACK, seq, ack, 12) != 0) break;
            } else if (type == T_CSR_WRITE && len == 8) {
                uint8_t p[8];
                if (read_exact(fd, p, 8) != 0) break;
                uint32_t a = get32(p), v = get32(p + 4);
                /* only registers that cannot desynchronise the ring: STATUS_CLR, THETA, SCRATCH (the daemon owns CONTROL, IN_BASE, IN_WR_SEQ) */
                if ((a & 3) == 0 && (a / 4 == R_STATUS_CLR || a / 4 == R_THETA || a / 4 == R_SCRATCH)) csr_wr(a / 4, v);
            } else if (type == T_CSR_READ && len == 4) {
                uint8_t p[4], r[8];
                if (read_exact(fd, p, 4) != 0) break;
                uint32_t a = get32(p);
                put32(r, a); put32(r + 4, (a < 256 * 4 && (a & 3) == 0) ? csr_rd(a / 4) : 0);
                if (send_msg(fd, T_CSR_REPLY, seq, r, 8) != 0) break;
            } else {
                send_error(fd, E_BAD_SIZE); break;
            }
        }
        /* completed frames: every step takes at least 8 ms and we poll every millisecond, so the snapshot registers still
         * belong to the frame whose DONE_SEQ we just saw */
        uint32_t ds = csr_rd(R_DONE_SEQ);
        while (out_done < ds) {
            unsigned slot = out_done % N_SLOTS;
            if (send_result(fd, seq_ring[slot]) != 0) goto done;
            out_done++;
        }
    }
done:
    engine_stop();
    free(inbuf);
}

int main(int argc, char **argv)
{
    int port = 5002;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ddr-base") && i + 1 < argc) ddr_phys = (uint32_t)strtoul(argv[++i], NULL, 0);
        else { fprintf(stderr, "usage: %s [--port N] [--ddr-base 0xADDR]\n", argv[0]); return 2; }
    }
    signal(SIGPIPE, SIG_IGN);                               /* a client vanishing must not kill the daemon */

#ifdef SNN_SIM
    ddr = (volatile uint8_t *)calloc(DDR_REGION_SPAN, 1);
    sim_regs[R_VERSION] = SNN_EXPECTED_BUILD; sim_regs[R_STATUS] = ST_READY;
    printf("snn_streamer (SIMULATED engine): fake results, port %d\n", port);
#else
    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) { fprintf(stderr, "open /dev/mem: %s (run as root)\n", strerror(errno)); return 1; }
    void *lw = mmap(NULL, HPS_LW_BRIDGE_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, (off_t)HPS_LW_BRIDGE_BASE);
    void *dr = mmap(NULL, DDR_REGION_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, (off_t)ddr_phys);
    if (lw == MAP_FAILED || dr == MAP_FAILED) { fprintf(stderr, "mmap: %s\n", strerror(errno)); return 1; }
    csr = (volatile uint32_t *)((uint8_t *)lw + SNN_CSR_OFFSET);
    ddr = (volatile uint8_t *)dr;
    printf("snn_streamer: CSR at 0x%08lX, VERSION 0x%08X, DDR3 ring at 0x%08X\n", HPS_LW_BRIDGE_BASE + SNN_CSR_OFFSET, csr_rd(R_VERSION), ddr_phys);
    /* The preloader on the SD card was built for an older design and leaves the FPGA-to-SDRAM ports in reset (fpgaportrst = 0):
     * the engine's reads return nothing (night-vision project, bug log 10). Release them here, exactly as nv_streamer does. */
    {
        volatile uint32_t *sdr = (volatile uint32_t *)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0xFFC25000);
        if (sdr == (volatile uint32_t *)MAP_FAILED) { fprintf(stderr, "mmap sdr: %s\n", strerror(errno)); return 1; }
        uint32_t before = sdr[0x80 / 4];
        if ((before & 1u) == 0) sdr[0x80 / 4] = 0x3FFF;
        printf("SDRAM fpgaportrst 0x%04X -> 0x%04X\n", before, sdr[0x80 / 4]);
    }
    /* bus echo test before anything else: a register that must read back what was written */
    csr_wr(R_SCRATCH, 0xA5C31E7Fu);
    if (csr_rd(R_SCRATCH) != 0xA5C31E7Fu) fprintf(stderr, "WARNING: CSR scratch echo failed (is the right bitstream loaded?)\n");
#endif

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0) { perror("bind"); return 1; }
    if (listen(srv, 2) != 0) { perror("listen"); return 1; }
    printf("listening on port %d (Ctrl+C to stop)\n", port);
    fflush(stdout);

    for (;;) {
        struct sockaddr_in cli;
        socklen_t clilen = sizeof(cli);
        int fd = accept(srv, (struct sockaddr *)&cli, &clilen);
        if (fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        printf("connection from %s\n", inet_ntoa(cli.sin_addr));
        fflush(stdout);
        handle_conn(fd);
        close(fd);
        printf("connection closed\n");
        fflush(stdout);
    }
    close(srv);
    return 0;
}
