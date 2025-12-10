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
#endif

using namespace taichi;

// ==========================================
// 1. Definitions and Constants
// ==========================================
using Vec = Vector3;
using Mat = Matrix3;

const int n = 64;           // Grid resolution
const real dt = 2e-4_f;     // Physical time step
const real frame_dt = 1.0_f / 60.0_f; // Frame time step
const real dx = 1.0_f / n;  // Grid spacing
const real inv_dx = 1.0_f / dx;

// Fluid parameters
const real particle_mass = 1.0_f;
const real vol = 1.0_f; 
const real K = 50.0_f;       
const real gamma_val = 7.0_f; 

struct Particle {
    Vec x, v;
    Mat C;       
    real J;      
    int c;       
    int type;    // 0: Water

    Particle(Vec x, int type = 0, int color = 0xFFFFFF) :
        x(x), v(Vec(0)), C(Mat(0)), J(1.0_f), type(type), c(color) {}
};

std::vector<Particle> particles;
Vector4 grid[n][n][n]; 

// ==========================================
// 2. Obstacle Motion Definition
// ==========================================
// The obstacle is a Box, doing circular motion on the ground.
// It does NOT rotate itself (orientation is fixed), only its position moves in a circle.

const Vec obs_size(0.05f, 0.1f, 0.05f); // Half-extents
const Vec motion_center(0.5f, 0.1f, 0.5f); // Center of the circular orbit (y is height)
const real motion_radius = 0.2f; // Radius of the circle
const real angular_speed = 3.0f; // Rad/s

// Helper to get current obstacle state
void get_obstacle_state(real time, Vec &position, Vec &velocity) {
    // x = cx + R * cos(omega * t)
    // z = cz + R * sin(omega * t)
    real theta = angular_speed * time;
    
    position.x = motion_center.x + motion_radius * std::cos(theta);
    position.z = motion_center.z + motion_radius * std::sin(theta);
    position.y = motion_center.y; // Keep height constant

    // v_x = -R * omega * sin(theta)
    // v_z =  R * omega * cos(theta)
    velocity.x = -motion_radius * angular_speed * std::sin(theta);
    velocity.z =  motion_radius * angular_speed * std::cos(theta);
    velocity.y = 0.0f;
}

// ==========================================
// 3. Core: Physical simulation
// ==========================================
void advance(real dt, real current_time) {
    // 1. Reset grid
    std::memset(grid, 0, sizeof(grid));

    // 2. [P2G] Particle -> Grid
    #pragma omp parallel for
    for (int i = 0; i < (int)particles.size(); i++) {
        auto& p = particles[i];
        // Note: particles array now ONLY contains water. Obstacle is handled analytically.

        Vector3i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();
        
        Vec w[3] = {
            Vec(0.5) * sqr(Vec(1.5) - fx),
            Vec(0.75) - sqr(fx - Vec(1.0)),
            Vec(0.5) * sqr(fx - Vec(0.5))
        };

        real effective_J = std::max(p.J, 0.1_f); 
        real pressure = K * (std::pow(effective_J, -gamma_val) - 1.0_f); 
        Mat stress = Mat(-pressure); 
        Mat affine = -(dt * vol) * (4 * inv_dx * inv_dx) * effective_J * stress + particle_mass * p.C; 

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

    // --- Calculate Obstacle State for this Substep ---
    Vec obs_pos, obs_v;
    get_obstacle_state(current_time, obs_pos, obs_v);
    
    // Update AABB for collision detection
    Vec obs_min = obs_pos - obs_size;
    Vec obs_max = obs_pos + obs_size;

    // 3. [Grid Update]
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                auto& g = grid[i][j][k];
                if (g.w > 1e-10_f) { 
                    g.x /= g.w; g.y /= g.w; g.z /= g.w;
                    g.y -= 9.8_f * dt; // Gravity

                    // --- Boundary condition (Container Walls) ---
                    int boundary = 2;
                    if (i < boundary && g.x < 0) g.x = 0;
                    if (i > n - boundary && g.x > 0) g.x = 0;
                    if (j < boundary && g.y < 0) g.y = 0;
                    if (j > n - boundary && g.y > 0) g.y = 0;
                    if (k < boundary && g.z < 0) g.z = 0;
                    if (k > n - boundary && g.z > 0) g.z = 0;

                    // --- Rigid body collision (Moving Obstacle) ---
                    Vec pos = Vec(i, j, k) * dx;

                    if (pos.x > obs_min.x && pos.x < obs_max.x &&
                        pos.y > obs_min.y && pos.y < obs_max.y &&
                        pos.z > obs_min.z && pos.z < obs_max.z) {
                        
                        // Calculate normal (closest face)
                        Vec dist_min = pos - obs_min;
                        Vec dist_max = obs_max - pos;
                        
                        real min_d = dist_min.x;
                        Vec normal(-1, 0, 0);

                        if (dist_max.x < min_d) { min_d = dist_max.x; normal = Vec(1, 0, 0); }
                        if (dist_min.y < min_d) { min_d = dist_min.y; normal = Vec(0, -1, 0); }
                        if (dist_max.y < min_d) { min_d = dist_max.y; normal = Vec(0, 1, 0); }
                        if (dist_min.z < min_d) { min_d = dist_min.z; normal = Vec(0, 0, -1); }
                        if (dist_max.z < min_d) { min_d = dist_max.z; normal = Vec(0, 0, 1); }

                        // [Key Change] Use Relative Velocity
                        Vec grid_v(g.x, g.y, g.z);
                        Vec v_rel = grid_v - obs_v; // Relative to moving obstacle

                        real v_n = dot(v_rel, normal);

                        // If fluid is moving INTO the obstacle (relative)
                        if (v_n < 0) {
                            Vec v_t = v_rel - normal * v_n;
                            // Friction: 0.0 = Sticky, 1.0 = Slip
                            Vec v_new_rel = v_t * 0.5f; 
                            
                            // Convert back to world velocity
                            Vec v_new = v_new_rel + obs_v;
                            
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

// Add water
void add_box(Vec center, Vec size, int type, int color) {
    real step = dx * 0.55f; 
    for (real x = center.x - size.x; x < center.x + size.x; x += step) {
        for (real y = center.y - size.y; y < center.y + size.y; y += step) {
            for (real z = center.z - size.z; z < center.z + size.z; z += step) {
                particles.push_back(Particle(Vec(x, y, z), type, color));
            }
        }
    }
}

// Save PLY with Dynamic Obstacle Visualization
void save_ply(int frame, real current_time) {
    char buffer[100];
    sprintf(buffer, "ply_moving_obstacle/output_%04d.ply", frame);
    FILE* f = fopen(buffer, "w");
    if (!f) {
        printf("Failed to open file: %s\n", buffer);
        return;
    }

    // 1. Generate obstacle visualization particles on the fly
    std::vector<Particle> obs_particles;
    Vec obs_pos, obs_v;
    get_obstacle_state(current_time, obs_pos, obs_v);
    
    // Temporary parameters for visualization density
    real step = dx * 0.55f; 
    Vec size = obs_size;
    
    for (real x = obs_pos.x - size.x; x < obs_pos.x + size.x; x += step) {
        for (real y = obs_pos.y - size.y; y < obs_pos.y + size.y; y += step) {
            for (real z = obs_pos.z - size.z; z < obs_pos.z + size.z; z += step) {
                obs_particles.push_back(Particle(Vec(x, y, z), 1, 0xFF3333));
            }
        }
    }

    int total_particles = particles.size() + obs_particles.size();
    
    fprintf(f, "ply\nformat ascii 1.0\n");
    fprintf(f, "element vertex %d\n", total_particles);
    fprintf(f, "property float x\nproperty float y\nproperty float z\n");
    fprintf(f, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
    fprintf(f, "end_header\n");

    // Write Water
    for (const auto& p : particles) {
        int r = (p.c >> 16) & 0xFF;
        int g = (p.c >> 8) & 0xFF;
        int b = p.c & 0xFF;
        fprintf(f, "%f %f %f %d %d %d\n", p.x.x, p.x.y, p.x.z, r, g, b);
    }
    // Write Obstacle
    for (const auto& p : obs_particles) {
        int r = (p.c >> 16) & 0xFF;
        int g = (p.c >> 8) & 0xFF;
        int b = p.c & 0xFF;
        fprintf(f, "%f %f %f %d %d %d\n", p.x.x, p.x.y, p.x.z, r, g, b);
    }

    fclose(f);
    printf("Frame %d saved. Water: %d, Obstacle: %d\n", frame, (int)particles.size(), (int)obs_particles.size());
}

int main() {
#ifdef _WIN32
    _mkdir("ply_moving_obstacle");
#else
    mkdir("ply_moving_obstacle", 0777);
#endif

    // 1. Water block initialization
    add_box(Vec(0.2, 0.5, 0.2), Vec(0.15, 0.45, 0.15), 0, 0x0000FF);
    
    // Note: We DO NOT add obstacle particles to the 'particles' vector anymore.
    // They are generated dynamically in save_ply for visualization.

    printf("Simulating...\n");
    int frame = 0;
    real current_time = 0.0f;

    while (frame < 300) {
        int substeps = (int)(frame_dt / dt);
        for (int i = 0; i < substeps; i++) {
            advance(dt, current_time);
            current_time += dt;
        }
        save_ply(frame++, current_time);
    }
    return 0;
}


/* ==========================================
   Cross-Platform Compilation Commands (ensure that taichi.h is in the same directory)
==========================================

1. Windows (MinGW/GCC):
   g++ dambreak_moving_obstacle.cpp -o dambreak_moving_obstacle.exe -std=c++14 -O3 -lgdi32 -fopenmp

2. Linux (Ubuntu/Debian etc.):
   # Requires: sudo apt-get install libx11-dev
   g++ dambreak_moving_obstacle.cpp -o dambreak_moving_obstacle -std=c++14 -O3 -lX11 -lpthread -fopenmp

3. macOS (Clang):
   # Requires: brew install libomp
   g++ dambreak_moving_obstacle.cpp -o dambreak_moving_obstacle -std=c++14 -O3 -Xpreprocessor -fopenmp -lomp -framework Cocoa -framework CoreGraphics

.\dambreak_moving_obstacle.exe   

*/
