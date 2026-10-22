#define _POSIX_C_SOURCE 199309L
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MAX_ITER 5000

typedef struct {
    const char *name;
    double cx, cy, span;
} View;

static const View VIEWS[] = {
    {"Full View",       -0.75,   0.0,    4.0},
    {"Seahorse Valley", -0.7453, 0.1127, 4.0 / 200.0},
    {"Elephant Valley",  0.2821, 0.0000, 4.0 / 50.0}
};

typedef struct {
    long long pixels;
    int max_diff;
    long long first;
} DiffResult;

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static const double STOPS[6] = {0.0, 0.16, 0.42, 0.6425, 0.8575, 1.0};
static const double PALETTE[6][3] = {
    {0, 7, 100},
    {32, 107, 203},
    {237, 255, 255},
    {255, 170, 0},
    {120, 20, 90},
    {0, 7, 100}
};

static void colour(int n, double x, double y, unsigned char *px) {
    if (n >= MAX_ITER) {
        px[0] = px[1] = px[2] = 0;
        return;
    }
    double mag2 = x * x + y * y;
    double mu = n + 1.0 - log2(0.5 * log(mag2) / log(2.0));
    if (mu < 0.0) mu = 0.0;
    double t = fmod(log(mu + 1.0) * 0.35, 1.0);
    int s = 0;
    while (s < 4 && t > STOPS[s + 1]) s++;
    double f = (t - STOPS[s]) / (STOPS[s + 1] - STOPS[s]);
    for (int c = 0; c < 3; c++)
        px[c] = (unsigned char)(PALETTE[s][c] + f * (PALETTE[s + 1][c] - PALETTE[s][c]) + 0.5);
}

static inline void render_row(unsigned char *img, int w, int py, double x0, double ci, double scale) {
    for (int px = 0; px < w; px++) {
        double cr = x0 + px * scale;
        double x = 0.0, y = 0.0, x2 = 0.0, y2 = 0.0;
        int n = 0;
        while (x2 + y2 < 4.0 && n < MAX_ITER) {
            y = 2.0 * x * y + ci;
            x = x2 - y2 + cr;
            x2 = x * x;
            y2 = y * y;
            n++;
        }
        colour(n, x, y, img + 3 * ((size_t)py * w + px));
    }
}

static void render_seq(unsigned char *img, int w, int h, const View *v) {
    double scale = v->span / w;
    double x0 = v->cx - 0.5 * v->span;
    double y0 = v->cy + 0.5 * scale * h;
    for (int py = 0; py < h; py++)
        render_row(img, w, py, x0, y0 - py * scale, scale);
}

static void render_par(unsigned char *img, int w, int h, const View *v) {
    double scale = v->span / w;
    double x0 = v->cx - 0.5 * v->span;
    double y0 = v->cy + 0.5 * scale * h;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int py = 0; py < h; py++)
        render_row(img, w, py, x0, y0 - py * scale, scale);
}

static DiffResult compare(const unsigned char *a, const unsigned char *b, unsigned char *mask, int w, int h) {
    long long count = 0, first = (long long)w * h;
    int maxd = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:count) reduction(max:maxd) reduction(min:first)
    for (int py = 0; py < h; py++) {
        for (int px = 0; px < w; px++) {
            long long i = (long long)py * w + px;
            int d = 0;
            for (int c = 0; c < 3; c++) {
                int e = abs((int)a[3 * i + c] - (int)b[3 * i + c]);
                if (e > d) d = e;
            }
            mask[i] = d ? 255 : 0;
            if (d) {
                count++;
                if (d > maxd) maxd = d;
                if (i < first) first = i;
            }
        }
    }
    DiffResult r = {count, maxd, first};
    return r;
}

int main(int argc, char **argv) {
    int w, h, view, threads;
    const char *prefix = "mandelbrot";

    if (argc >= 5) {
        w = atoi(argv[1]);
        h = atoi(argv[2]);
        view = atoi(argv[3]);
        threads = atoi(argv[4]);
        if (argc >= 6) prefix = argv[5];
    } else {
        printf("Width: ");
        if (scanf("%d", &w) != 1) return 1;
        printf("Height: ");
        if (scanf("%d", &h) != 1) return 1;
        printf("View (1 = Full, 2 = Seahorse Valley, 3 = Elephant Valley): ");
        if (scanf("%d", &view) != 1) return 1;
        printf("Threads (%d cores available): ", omp_get_num_procs());
        if (scanf("%d", &threads) != 1) return 1;
    }

    if (w <= 0 || h <= 0 || view < 1 || view > 3 || threads < 1) {
        fprintf(stderr, "Usage: %s <width> <height> <view 1-3> <threads> [output prefix]\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(threads);

    const View *v = &VIEWS[view - 1];
    long long n = (long long)w * h;
    unsigned char *seq = malloc((size_t)n * 3);
    unsigned char *par = malloc((size_t)n * 3);
    unsigned char *mask = malloc((size_t)n);
    if (!seq || !par || !mask) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }

    char seq_file[512], par_file[512], diff_file[512];
    snprintf(seq_file, sizeof seq_file, "%s_seq.png", prefix);
    snprintf(par_file, sizeof par_file, "%s_par.png", prefix);
    snprintf(diff_file, sizeof diff_file, "%s_diff.png", prefix);

    double t0 = now_sec();
    render_seq(seq, w, h, v);
    double t1 = now_sec();
    stbi_write_png(seq_file, w, h, 3, seq, w * 3);
    double t2 = now_sec();

    double t3 = now_sec();
    render_par(par, w, h, v);
    double t4 = now_sec();
    stbi_write_png(par_file, w, h, 3, par, w * 3);
    double t5 = now_sec();

    free(seq);
    free(par);

    int wa = 0, ha = 0, ca = 0, wb = 0, hb = 0, cb = 0;
    unsigned char *img_a = NULL, *img_b = NULL;
    double t6 = now_sec();
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        img_a = stbi_load(seq_file, &wa, &ha, &ca, 3);
        #pragma omp section
        img_b = stbi_load(par_file, &wb, &hb, &cb, 3);
    }
    double t7 = now_sec();

    if (!img_a || !img_b || wa != w || ha != h || wb != w || hb != h) {
        fprintf(stderr, "Failed to read back images for the pixel check\n");
        return 1;
    }

    DiffResult r = compare(img_a, img_b, mask, w, h);
    double t8 = now_sec();
    stbi_write_png(diff_file, w, h, 1, mask, w);

    double seq_c = (t1 - t0) * 1e3, seq_p = (t2 - t1) * 1e3;
    double par_c = (t4 - t3) * 1e3, par_p = (t5 - t4) * 1e3;
    double speedup = seq_c / par_c;

    printf("\n%s | %dx%d | %d iterations | %d threads | schedule dynamic,1\n\n",
           v->name, w, h, MAX_ITER, threads);
    printf("%-10s %14s %14s %14s\n", "", "Compute (ms)", "PNG (ms)", "Total (ms)");
    printf("%-10s %14.2f %14.2f %14.2f\n", "Serial", seq_c, seq_p, seq_c + seq_p);
    printf("%-10s %14.2f %14.2f %14.2f\n", "Parallel", par_c, par_p, par_c + par_p);
    printf("\nSpeedup (compute): %.2fx\n", speedup);
    printf("Speedup (end to end): %.2fx\n", (seq_c + seq_p) / (par_c + par_p));

    printf("\nPixel check (%d threads): read %.2f ms, compare %.2f ms\n",
           threads, (t7 - t6) * 1e3, (t8 - t7) * 1e3);
    printf("Differing pixels: %lld of %lld (%.4f%%)\n", r.pixels, n, 100.0 * r.pixels / n);
    printf("Max channel difference: %d\n", r.max_diff);
    if (r.pixels) {
        long long i = r.first;
        printf("First difference at (x=%lld, y=%lld): serial (%d,%d,%d) vs parallel (%d,%d,%d)\n",
               i % w, i / w,
               img_a[3 * i], img_a[3 * i + 1], img_a[3 * i + 2],
               img_b[3 * i], img_b[3 * i + 1], img_b[3 * i + 2]);
        printf("Result: FAIL\n");
    } else {
        printf("Result: PASS, images are identical\n");
    }

    printf("\nImages: %s, %s, %s\n", seq_file, par_file, diff_file);

    stbi_image_free(img_a);
    stbi_image_free(img_b);
    free(mask);
    return r.pixels ? 2 : 0;
}
