#define _CRT_SECURE_NO_WARNINGS
#include "taichi.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <omp.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

using namespace taichi;

// ==========================================
// 1. Definitions and Constants
// ==========================================
using Vec = Vector3;
using Mat = Matrix3;

const int n = 64;           // Grid resolution (Number of cells per side, not number of grid nodes)
const real dt = 2e-4_f;     // Physical time step
const real frame_dt = 1.0_f / 60.0_f; // Frame time step (60 FPS)
const real dx = 1.0_f / n;  // Grid spacing
const real inv_dx = 1.0_f / dx;

// Fluid parameters (Don't consider viscosity for now)
const real particle_mass = 1.0_f;
const real vol = 1.0_f; 
// Parameters for Tait-Murnaghan Equation of water:     
const real K = 50.0_f;       
const real gamma_val = 7.0_f; 

struct Particle {
    Vec x, v;
    Mat C;       
    real J;      // Jacobian of deformation gradient (Only consider elastic deformation)
    int c;       // Color
    int type;    // 0: Water, 1: Rigid obstacle

    Particle(Vec x, int type = 0, int color = 0xFFFFFF) :
        x(x), v(Vec(0)), C(Mat(0)), J(1.0_f), type(type), c(color) {}
};

std::vector<Particle> particles;
Vector4 grid[n][n][n]; // [velocity_x, velocity_y, velocity_z, mass] (Grid nodes, not cells)

// ==========================================
// Obstacle Definition (Red Rigid Column)
// ==========================================
// Location of canter and half-extent
const Vec obs_center(0.45f, 0.1f, 0.45f); // This is the real physical position (X, Y, Z) (Refer to the Paraview rendering results for better understanding)
const Vec obs_size(0.05f, 0.1f, 0.05f);
const Vec obs_min = obs_center - obs_size;
const Vec obs_max = obs_center + obs_size;

// ==========================================
// 2. Physical simulation
// ==========================================
void advance(real dt) {
    // 1. Reset grid
    std::memset(grid, 0, sizeof(grid));

    // 2. [P2G] Particle -> Grid
    #pragma omp parallel for
    for (int i = 0; i < (int)particles.size(); i++) {
        auto& p = particles[i];
        if (p.type == 1) continue; 

        Vector3i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();
        
        // Quadratic B-spline weights
        Vec w[3] = {
            Vec(0.5) * sqr(Vec(1.5) - fx),
            Vec(0.75) - sqr(fx - Vec(1.0)),
            Vec(0.5) * sqr(fx - Vec(0.5))
        };

        real effective_J = std::max(p.J, 0.1_f); 
        real pressure = K * (std::pow(effective_J, -gamma_val) - 1.0_f); // Tait-Murnaghan Equation of State
        Mat stress = Mat(-pressure); // Neglect viscosity for now
        Mat affine = -(dt * vol) * (4 * inv_dx * inv_dx) * effective_J * stress + particle_mass * p.C; // MLS-MPM affine term

        for (int ix = 0; ix < 3; ix++) {
            for (int iy = 0; iy < 3; iy++) {
                for (int iz = 0; iz < 3; iz++) {
                    Vec dpos = (Vec(ix, iy, iz) - fx) * dx;
                    Vector4 contrib(p.v * particle_mass + affine * dpos, particle_mass);
                    real weight = w[ix].x * w[iy].y * w[iz].z;
                    
                    int idx_x = base_coord.x + ix;
                    int idx_y = base_coord.y + iy;
                    int idx_z = base_coord.z + iz;

                    if (idx_x >= 0 && idx_x < n && idx_y >= 0 && idx_y < n && idx_z >= 0 && idx_z < n) {
                        #pragma omp atomic
                        grid[idx_x][idx_y][idx_z].x += weight * contrib.x;
                        #pragma omp atomic
                        grid[idx_x][idx_y][idx_z].y += weight * contrib.y;
                        #pragma omp atomic
                        grid[idx_x][idx_y][idx_z].z += weight * contrib.z;
                        #pragma omp atomic
                        grid[idx_x][idx_y][idx_z].w += weight * contrib.w;
                    }
                }
            }
        }
    }

    // 3. [Grid Update] Grid force update + Boundary condition + Rigid body collision
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                auto& g = grid[i][j][k];
                if (g.w > 1e-10_f) { 
                    g.x /= g.w; g.y /= g.w; g.z /= g.w;
                    g.y -= 9.8_f * dt; // Gravity

                    // --- Boundary condition ---
                    int boundary = 2; // Thickness of wall
                    if (i < boundary && g.x < 0) g.x = 0;
                    if (i > n - boundary && g.x > 0) g.x = 0;
                    if (j < boundary && g.y < 0) g.y = 0;
                    if (j > n - boundary && g.y > 0) g.y = 0;
                    if (k < boundary && g.z < 0) g.z = 0;
                    if (k > n - boundary && g.z > 0) g.z = 0;

                    // --- Rigid body collision ---
                    Vec pos = Vec(i, j, k) * dx; // (Real) Physical position of the grid node

                    // Determine whether the grid not is inside the rigid body
                    if (pos.x > obs_min.x && pos.x < obs_max.x &&
                        pos.y > obs_min.y && pos.y < obs_max.y &&
                        pos.z > obs_min.z && pos.z < obs_max.z) {
                        
                        // Calculate the distance from the point to each face and find the closest face as the normal direction
                        Vec dist_min = pos - obs_min;
                        Vec dist_max = obs_max - pos;
                        
                        // Find the axis corresponding to the minimum penetration depth
                        real min_d = dist_min.x;
                        Vec normal(-1, 0, 0); // Left face as default

                        if (dist_max.x < min_d) { min_d = dist_max.x; normal = Vec(1, 0, 0); }
                        if (dist_min.y < min_d) { min_d = dist_min.y; normal = Vec(0, -1, 0); }
                        if (dist_max.y < min_d) { min_d = dist_max.y; normal = Vec(0, 1, 0); }
                        if (dist_min.z < min_d) { min_d = dist_min.z; normal = Vec(0, 0, -1); }
                        if (dist_max.z < min_d) { min_d = dist_max.z; normal = Vec(0, 0, 1); }

                        // Relative velocity (since the column is stationary, it is just grid.v)
                        Vec v_rel = Vec(g.x, g.y, g.z);
                        real v_n = dot(v_rel, normal);

                        // If the fluid is colliding into the inside of the column (v_n < 0), then handle the collision
                        if (v_n < 0) {
                            Vec v_t = v_rel - normal * v_n;
                            // Friction force handling: Set to 0.5 (semi-sliding, semi-sticking)
                            // If set to 0, completely viscous; if set to 1, completely slipping  
                            Vec v_new = v_t * 0.5f; 
                            
                            g.x = v_new.x;
                            g.y = v_new.y;
                            g.z = v_new.z;
                        }
                    }
                }
            }
        }
    }

    // 4. [G2P] Grid -> Particle
    #pragma omp parallel for
    for (int i = 0; i < (int)particles.size(); i++) {
        auto& p = particles[i];
        if (p.type == 1) continue; 

        Vector3i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();
        Vec w[3] = {
            Vec(0.5) * sqr(Vec(1.5) - fx),
            Vec(0.75) - sqr(fx - Vec(1.0)),
            Vec(0.5) * sqr(fx - Vec(0.5))
        };

        p.v = Vec(0);
        p.C = Mat(0);

        for (int ix = 0; ix < 3; ix++) {
            for (int iy = 0; iy < 3; iy++) {
                for (int iz = 0; iz < 3; iz++) {
                    Vec dpos = (Vec(ix, iy, iz) - fx);
                    int idx_x = base_coord.x + ix;
                    int idx_y = base_coord.y + iy;
                    int idx_z = base_coord.z + iz;

                    if (idx_x >= 0 && idx_x < n && idx_y >= 0 && idx_y < n && idx_z >= 0 && idx_z < n) {
                        Vector4 g_val = grid[idx_x][idx_y][idx_z];
                        Vec grid_v(g_val.x, g_val.y, g_val.z);
                        real weight = w[ix].x * w[iy].y * w[iz].z;

                        p.v += weight * grid_v;
                        p.C += 4 * inv_dx * Mat::outer_product(weight * grid_v, dpos);
                    }
                }
            }
        }

        p.x += dt * p.v;
        p.J *= (1.0_f + dt * (p.C[0][0] + p.C[1][1] + p.C[2][2]));

        p.x.x = taichi::clamp(p.x.x, 1.0_f * dx, (n - 1.0_f) * dx);
        p.x.y = taichi::clamp(p.x.y, 1.0_f * dx, (n - 1.0_f) * dx);
        p.x.z = taichi::clamp(p.x.z, 1.0_f * dx, (n - 1.0_f) * dx);
    }
}

// Add a box of particles (type: 0 for water, 1 for rigid obstacle)
void add_box(Vec center, Vec size, int type, int color, Vec initial_velocity = Vec(0)) {
    real step = dx * 0.55f; 
    for (real x = center.x - size.x; x < center.x + size.x; x += step) {
        for (real y = center.y - size.y; y < center.y + size.y; y += step) {
            for (real z = center.z - size.z; z < center.z + size.z; z += step) {
                Particle p(Vec(x, y, z), type, color);
                p.v = initial_velocity; // Optional initial velocity (Only for water particles)
                particles.push_back(p);
            }
        }
    }
}

void save_ply(int frame) {
    char buffer[100];
    sprintf(buffer, "ply_stationary_obstacle/output_%04d.ply", frame);
    FILE* f = fopen(buffer, "w");
    if (!f) {
        printf("Failed to open file: %s\n", buffer);
        return;
    }
    
    fprintf(f, "ply\nformat ascii 1.0\n");
    fprintf(f, "element vertex %d\n", (int)particles.size());
    fprintf(f, "property float x\nproperty float y\nproperty float z\n");
    fprintf(f, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
    fprintf(f, "end_header\n");

    for (const auto& p : particles) {
        int r = (p.c >> 16) & 0xFF;
        int g = (p.c >> 8) & 0xFF;
        int b = p.c & 0xFF;
        fprintf(f, "%f %f %f %d %d %d\n", p.x.x, p.x.y, p.x.z, r, g, b);
    }
    fclose(f);
    printf("Frame %d saved. Particles: %d\n", frame, (int)particles.size());
}

int main() {
#ifdef _WIN32
    _mkdir("ply_stationary_obstacle");
#else
    mkdir("ply_stationary_obstacle", 0777);
#endif

    // 1. Water block initialization (Blue, type=0)
    add_box(Vec(0.2, 0.5, 0.2), Vec(0.15, 0.45, 0.15), 0, 0x0000FF, Vec(0, 0, 0));

    // 2. Rigid obstacle initialization (Red, type=1)
    add_box(obs_center, obs_size, 1, 0xFF3333);

    printf("Simulating with %d particles...\n", (int)particles.size());
    int frame = 0;
    while (frame < 300) {
        int substeps = (int)(frame_dt / dt);
        for (int i = 0; i < substeps; i++) {
            advance(dt);
        }
        save_ply(frame++);
    }
    return 0;
}




/* ==========================================
   Cross-Platform Compilation Commands (ensure that taichi.h is in the same directory)
==========================================

1. Windows (MinGW/GCC):
   g++ dambreak_stationary_obstacle.cpp -o dambreak_stationary_obstacle.exe -std=c++14 -O3 -lgdi32 -fopenmp

2. Linux (Ubuntu/Debian etc.):
   # Requires: sudo apt-get install libx11-dev
   g++ dambreak_stationary_obstacle.cpp -o dambreak_stationary_obstacle -std=c++14 -O3 -lX11 -lpthread -fopenmp

3. macOS (Clang):
   # Requires: brew install libomp
   g++ dambreak_stationary_obstacle.cpp -o dambreak_stationary_obstacle -std=c++14 -O3 -Xpreprocessor -fopenmp -lomp -framework Cocoa -framework CoreGraphics

.\dambreak_stationary_obstacle.exe   

   */

 