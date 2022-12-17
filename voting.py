import cupy as cp

vote_kernel = cp.RawKernel(r'''
    #include "/home/neil/ppf_matching/helper_math.cuh"
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void vote_kernel(
        const float *point_xys, 
        const float *point_offsets, 
        const float *point_sizes,
        const int *pair_idxs,
        int n_pairs,
        float *grid_obj,
        float *grid_size,
        float *grid_cnt,
        int grid_x, 
        int grid_y,
        int width,
        int height,
        float grid_intvl,
        const float *point_weights
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_pairs) {
            int idx1 = pair_idxs[idx * 2];
            int idx2 = pair_idxs[idx * 2 + 1];
            float2 d1 = make_float2(point_offsets[idx1 * 2], point_offsets[idx1 * 2 + 1]);
            float2 d2 = make_float2(point_offsets[idx2 * 2], point_offsets[idx2 * 2 + 1]);
            
            float2 xy1 = make_float2(point_xys[idx1 * 2], point_xys[idx1 * 2 + 1]) / height;
            float2 xy2 = make_float2(point_xys[idx2 * 2], point_xys[idx2 * 2 + 1]) / height;
            
            float ox2ox1 = xy2.x - xy1.x;
            float oy2oy1 = xy2.y - xy1.y;
            
            float denom = -d2.x * d1.y + d2.y * d1.x;
            
            if (abs(denom) < 1e-9) return;
            float a = (ox2ox1 * d2.y - oy2oy1 * d2.x) / denom;
            if (a <= 0) return;
            
            float prob = point_weights[idx1] * point_weights[idx2];
            float2 center_grid = (xy1 + a * d1) * height;
            
            xy1 *= height;
            xy2 *= height;
            
            center_grid /= grid_intvl;
        
            if (center_grid.x < 0.01 || center_grid.y < 0.01 || 
                center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01) {
                return;
            }
            int2 center_grid_floor = make_int2(center_grid);
            int2 center_grid_ceil = center_grid_floor + 1;
            float2 residual = fracf(center_grid);
            
            float2 w0 = 1.f - residual;
            float2 w1 = residual;
            
            float ll = w0.x * w0.y;
            float lh = w0.x * w1.y;
            float hl = w1.x * w0.y;
            float hh = w1.x * w1.y;
            
            center_grid *= grid_intvl;
            
            float thresh = 0.5f;
            if (point_sizes[idx1 * 2] > thresh && point_sizes[idx1 * 2] > thresh && point_sizes[idx2 * 2] > thresh && point_sizes[idx2 * 2] > thresh) {
                float size_x = (abs(center_grid.x - xy1.x) / height / point_sizes[idx1 * 2] + abs(center_grid.x - xy2.x) / height / point_sizes[idx2 * 2]) / 2.f;
                float size_y = (abs(center_grid.y - xy1.y) / height / point_sizes[idx1 * 2 + 1] + abs(center_grid.y - xy2.y) / height / point_sizes[idx2 * 2 + 1]) / 2.f;
                atomicAdd(&grid_size[(center_grid_floor.x * grid_y + center_grid_floor.y) * 2], size_x * prob);
                atomicAdd(&grid_size[(center_grid_floor.x * grid_y + center_grid_floor.y) * 2 + 1], size_y * prob);
                atomicAdd(&grid_cnt[center_grid_floor.x * grid_y + center_grid_floor.y], prob);
            }
            
            atomicAdd(&grid_obj[center_grid_floor.x * grid_y + center_grid_floor.y], ll * prob);
            atomicAdd(&grid_obj[center_grid_floor.x * grid_y + center_grid_ceil.y], lh * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y + center_grid_floor.y], hl * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y + center_grid_ceil.y], hh * prob);
        }
    }
''', 'vote_kernel')