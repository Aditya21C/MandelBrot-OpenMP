#define _POSIX_C_SOURCE 199309L
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
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
    {"Elephant Valley",  0.3300, 0.0500, 4.0 / 50.0}
};

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

static void render(unsigned char *img, int w, int h, const View *v) {
    double scale = v->span / w;
    double x0 = v->cx - 0.5 * v->span;
    double y0 = v->cy + 0.5 * scale * h;
    #pragma omp parallel for schedule(runtime)
    for (int py = 0; py < h; py++) {
        double ci = y0 - py * scale;
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
}

int main(int argc, char **argv) {
    int w, h, view;
    const char *out = "mandelbrot_par.png";

    if (argc >= 4) {
        w = atoi(argv[1]);
        h = atoi(argv[2]);
        view = atoi(argv[3]);
        if (argc >= 5) out = argv[4];
    } else {
        printf("Width: ");
        if (scanf("%d", &w) != 1) return 1;
        printf("Height: ");
        if (scanf("%d", &h) != 1) return 1;
        printf("View (1 = Full, 2 = Seahorse Valley, 3 = Elephant Valley): ");
        if (scanf("%d", &view) != 1) return 1;
    }

    if (w <= 0 || h <= 0 || view < 1 || view > 3) {
        fprintf(stderr, "Usage: %s <width> <height> <view 1-3> [output.png]\n", argv[0]);
        return 1;
    }

    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_dynamic, 1);

    const View *v = &VIEWS[view - 1];
    unsigned char *img = malloc((size_t)w * h * 3);
    if (!img) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }

    double t0 = now_sec();
    render(img, w, h, v);
    double t1 = now_sec();
    stbi_write_png(out, w, h, 3, img, w * 3);
    double t2 = now_sec();

    omp_sched_t kind;
    int chunk;
    omp_get_schedule(&kind, &chunk);
    const char *kinds[] = {"?", "static", "dynamic", "guided", "auto"};
    int k = (int)kind & 0x7fffffff;
    printf("OpenMP | %d threads | schedule %s,%d | %s | %dx%d | %d iter\n",
           omp_get_max_threads(), (k >= 1 && k <= 4) ? kinds[k] : "?", chunk,
           v->name, w, h, MAX_ITER);
    printf("Compute: %.2f ms\nPNG write: %.2f ms\nTotal: %.2f ms\nOutput: %s\n",
           (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3, out);

    free(img);
    return 0;
}
