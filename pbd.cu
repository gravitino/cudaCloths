
#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <cstring>
#include <vector>

#include <omp.h>

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

// error makro
#define CUERR {                                                              \
    cudaError_t err;                                                         \
    if ((err = cudaGetLastError()) != cudaSuccess) {                         \
       std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "       \
                 << __FILE__ << ", line " << __LINE__ << std::endl;          \
       exit(1);                                                              \
    }                                                                        \
}

#define HIQ

#ifndef HIQ

#define N (64)
#define EPS (1E-2)
#define BIAS (0.17)
#define ITERS (128)
#define G (1)

#else

#define N (128)
#define EPS (1E-2)
#define BIAS (0.15)
#define ITERS (400)
#define G (1)

#endif


/**************************************************
 * constants
 **************************************************/

const unsigned int window_width = 1920;
const unsigned int window_height = 1080;

// vbo
static GLuint ibo = 0;
static GLuint vbo = 0;
static GLuint cbo = 0;
static GLuint nbo = 0;

// host-sided vectors
static std::vector<float> vertices;
static std::vector<float> velocities;
static std::vector<float> colors;
static std::vector<float> normals;
static std::vector<int> indices;

// device-sided vectors
struct cudaGraphicsResource *vbo_resource;
struct cudaGraphicsResource *vbo_resource2;
void *g_vbo_buffer;
void *g_vbo_buffer2;
float3 * Velocities;
float3 * Temp;

// constraints
static float cnstr_two;
static float cnstr_dia;

// frames
static size_t frames = 0;

/**************************************************
 * forwards
 **************************************************/

void display();
void init_GL();
void stop_GL();
void init_data(int);

/**************************************************
 * GL code
 **************************************************/

__global__
void copy_kernel(float3 * target, float3 * source, int n) {

    const int j = blockDim.x*blockIdx.x+threadIdx.x;
    const int i = blockDim.y*blockIdx.y+threadIdx.y;

    if (i < n && j < n)
        target[i*n+j] = source[i*n+j];
}

///////////////////////////////////////////////////////////////////////////////
// helper
///////////////////////////////////////////////////////////////////////////////

__host__ __device__ __forceinline__
void update_positions(float3& pos, float3& vel, const float eps) {

    // increase velocity if outside of the sphere else set 0
    vel.z -= (pos.x*pos.x+pos.y*pos.y+pos.z*pos.z > 1) ? eps*G : 0;

    // x = x-eps*v  <=> v = delta s / delta t
    pos.x += eps*vel.x;
    pos.y += eps*vel.y;
    pos.z += eps*vel.z;
}

__host__ __device__ __forceinline__
void adjust_positions(float3& pos) {

    // inverse radius
    const float invrho = rsqrtf(pos.x*pos.x+pos.y*pos.y+pos.z*pos.z);

    // move vertex to surface of sphere if inside
    pos.x *= invrho < 1 ? 1 : invrho;
    pos.y *= invrho < 1 ? 1 : invrho;
    pos.z *= invrho < 1 ? 1 : invrho;
}

__host__ __device__ __forceinline__
void relax_constraint(const float3 * Pos, float3 * Tmp,
                      const int l, const int m,
                      const float constraint, const float bias) {

    // displacement vector
    float3 delta;
    delta.x = Pos[l].x-Pos[m].x;
    delta.y = Pos[l].y-Pos[m].y;
    delta.z = Pos[l].z-Pos[m].z;

    // inverse length of the displacement vector
    const float invlen = rsqrtf(delta.x*delta.x+
                                delta.y*delta.y+
                                delta.z*delta.z);

    // this is exactly zero if length == constraint
    const float factor = (1.0f-constraint*invlen)*bias;

    // update the coordinates

    #if defined(__CUDA_ARCH__)
		atomicAdd(&Tmp[l].x, -delta.x*factor);
        atomicAdd(&Tmp[l].y, -delta.y*factor);
        atomicAdd(&Tmp[l].z, -delta.z*factor);

        atomicAdd(&Tmp[m].x, +delta.x*factor);
        atomicAdd(&Tmp[m].y, +delta.y*factor);
        atomicAdd(&Tmp[m].z, +delta.z*factor);
    #else
        //# pragma omp atomic
        Tmp[l].x -= delta.x*factor;
        //# pragma omp atomic
        Tmp[l].y -= delta.y*factor;
        //# pragma omp atomic
        Tmp[l].z -= delta.z*factor;

        //# pragma omp atomic
        Tmp[m].x += delta.x*factor;
        //# pragma omp atomic
        Tmp[m].y += delta.y*factor;
        //# pragma omp atomic
        Tmp[m].z += delta.z*factor;
    #endif
}

__host__ __device__ __forceinline__
void normalize(float3& normal) {

    const float invrho = rsqrt(normal.x*normal.x+
                               normal.y*normal.y+
                               normal.z*normal.z);

    normal.x *= invrho;
    normal.y *= invrho;
    normal.z *= invrho;
}

__host__ __device__ __forceinline__
void wedge(const float3 * Vertices, float3& Normal,
           const int& i, const int& j, const int n,
           const int& a, const int& b) {

    // three points spanning a triangle
    float3 center = Vertices[i*n+j];
    float3 span_u = Vertices[(i+a)*n+j];
    float3 span_v = Vertices[i*n+(j+b)];

    // first spanning vector
    span_u.x -= center.x;
    span_u.y -= center.y;
    span_u.z -= center.z;

    // second spanning vectors
    span_v.x -= center.x;
    span_v.y -= center.y;
    span_v.z -= center.z;

    // cross product of span_u and span_v
    float3 cross;
    cross.x = span_u.y*span_v.z-span_v.y*span_u.z;
    cross.y = span_u.z*span_v.x-span_v.z*span_u.x;
    cross.z = span_u.x*span_v.y-span_v.x*span_u.y;

    // add to normal information
    Normal.x += cross.x*a*b;
    Normal.y += cross.y*a*b;
    Normal.z += cross.z*a*b;
}

//////////////////////////////////////////////////////////////////////////////
// kernels
//////////////////////////////////////////////////////////////////////////////

__global__
void propagate_kernel(float3 * Vertices, float3 * Velocities,
                      const int n, const float eps=EPS) {

    const int j = blockDim.x*blockIdx.x+threadIdx.x;
    const int i = blockDim.y*blockIdx.y+threadIdx.y;

    if (i < n && j < n)
        update_positions(Vertices[i*n+j], Velocities[i*n+j], eps);
}

__global__
void validate_kernel(float3 * Vertices, float3 * Temp,
                     const float cnstr_two, const float cnstr_dia,
                     const int n, const float bias=BIAS) {

    const int j = blockDim.x*blockIdx.x+threadIdx.x;
    const int i = blockDim.y*blockIdx.y+threadIdx.y;


    if (i < n-1 && j < n)
        relax_constraint(Vertices, Temp,
                         i*n+j, (i+1)*n+j, cnstr_two, bias);
    if (i < n && j < n-1)
        relax_constraint(Vertices, Temp,
                         i*n+j, i*n+(j+1), cnstr_two, bias);
    if (i < n-2 && j < n)
        relax_constraint(Vertices, Temp,
                         i*n+j, (i+2)*n+j, 2*cnstr_two, bias);
    if (i < n && j < n-2)
        relax_constraint(Vertices, Temp,
                         i*n+j, i*n+(j+2), 2*cnstr_two, bias);
    if (i < n-1 && j < n-1)
        relax_constraint(Vertices, Temp,
                         i*n+j, (i+1)*n+(j+1), cnstr_dia, bias);

    if (i > 0 && i < n && j < n-1)
        relax_constraint(Vertices, Temp,
                         i*n+j, (i-1)*n+(j+1), cnstr_dia, bias);
}

__global__
void adjust_kernel(float3* Temp, const int n) {
    const int j = blockDim.x*blockIdx.x+threadIdx.x;
    const int i = blockDim.y*blockIdx.y+threadIdx.y;

    if (i < n && j < n)
        adjust_positions(Temp[i*n+j]);
}

__global__
void update_normals_kernel(float3 * Vertices, float3 * Normals, const int n) {

    const int j = blockDim.x*blockIdx.x+threadIdx.x;
    const int i = blockDim.y*blockIdx.y+threadIdx.y;

    if (i < n && j < n) {

        // zeros normal
        float3 Normal;
        Normal.x = 0;
        Normal.y = 0;
        Normal.z = 0;

        if (i > 0 && j > 0)
            wedge(Vertices, Normal, i, j, n, -1, -1);
        if (i > 0 && j+1 < n)
            wedge(Vertices, Normal, i, j, n, -1, +1);
        if (i+1 < n && j > 0)
            wedge(Vertices, Normal, i, j, n, +1, -1);
        if (i+1 < n && j+1 < n)
            wedge(Vertices, Normal, i, j, n, +1, +1);

        // normalize normal
        normalize(Normal);

        // store result
        Normals[i*n+j] = Normal;
    }

}

void propagate_gpu(int n) {

    float3 * Vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&Vertices, &num_bytes, vbo_resource);

    dim3 grid((n+7)/8, (n+7)/8, 1);
    dim3 blck(8, 8, 1);

    propagate_kernel<<<grid, blck>>>(Vertices, Velocities, n);            CUERR

    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

void validate_gpu(int n, const int iters=ITERS) {

    float3 * Vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&Vertices, &num_bytes, vbo_resource);

    dim3 grid((n+7)/8, (n+7)/8, 1);
    dim3 blck(8, 8, 1);

    for (int iter = 0; iter < iters; iter++) {
        copy_kernel<<<grid, blck>>>(Temp, Vertices, n);
        validate_kernel<<<grid, blck>>>(Vertices, Temp, cnstr_two, cnstr_dia, n);
        adjust_kernel<<<grid, blck>>>(Temp, n);
        copy_kernel<<<grid, blck>>>(Vertices, Temp, n);
    }

    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

void update_normals_gpu(int n) {

    float3 * Vertices, * Normals;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&Vertices, &num_bytes, vbo_resource);
    cudaGraphicsMapResources(1, &vbo_resource2, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&Normals, &num_bytes, vbo_resource2);

    dim3 grid((n+7)/8, (n+7)/8, 1);
    dim3 blck(8, 8, 1);

    update_normals_kernel<<<grid, blck>>>(Vertices, Normals, n); CUERR

    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
    cudaGraphicsUnmapResources(1, &vbo_resource2, 0);

}

void propagate(float3 * vertices, float3 * velocities,
               const int n, const float eps=EPS) {

	// let gravitiy do its job
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            update_positions(vertices[i*n+j], velocities[i*n+j], eps);
}

void validate(float3 * vertices, const int n,
              const int iters=ITERS, const float bias=BIAS) {

    // double buffering
    std::vector<float3> temp(3*n*n);

    #pragma omp parallel
    for (int iter = 0; iter < iters; iter++) {

        #pragma omp single
        std::memcpy(temp.data(), vertices, sizeof(float3)*n*n);


        # pragma omp for nowait
        for (size_t i = 0; i < n-1; i++)
            for (size_t j = 0; j < n; j++)
                relax_constraint(vertices, temp.data(),
                                 i*n+j, (i+1)*n+j, cnstr_two, bias);

        # pragma omp for nowait
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n-1; j++)
                relax_constraint(vertices, temp.data(),
                                 i*n+j, i*n+(j+1), cnstr_two, bias);

        # pragma omp for nowait
        for (size_t i = 0; i < n-2; i++)
            for (size_t j = 0; j < n; j++)
                relax_constraint(vertices, temp.data(),
                                 i*n+j, (i+2)*n+j, 2*cnstr_two, bias);

        # pragma omp for nowait
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n-2; j++)
                relax_constraint(vertices, temp.data(),
                                  i*n+j, i*n+(j+2), 2*cnstr_two, bias);

        # pragma omp for nowait
        for (size_t i = 0; i < n-1; i++)
            for (size_t j = 0; j < n-1; j++)
                relax_constraint(vertices, temp.data(),
                                 i*n+j, (i+1)*n+(j+1), cnstr_dia, bias);

        # pragma omp for nowait
        for (size_t i = 1; i < n; i++)
            for (size_t j = 0; j < n-1; j++)
                relax_constraint(vertices, temp.data(),
                                  i*n+j, (i-1)*n+(j+1), cnstr_dia, bias);

        # pragma omp for
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                adjust_positions(temp[i*n+j]);


        # pragma omp single
        std::memcpy(vertices, temp.data(), sizeof(float3)*n*n);
    }

}

void update_normals(float3 * vertices,
                    float3 * normals,
                    const int n) {

    // set normals to zero
    std::memset(normals, 0, sizeof(float3)*n*n);

    # pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {

            float3 normal;
            normal.x = 0;
            normal.y = 0;
            normal.z = 0;

            if (i > 0 && j > 0)
                wedge(vertices, normal, i, j, n, -1, -1);
            if (i > 0 && j+1 < n)
                wedge(vertices, normal, i, j, n, -1, +1);
            if (i+1 < n && j > 0)
                wedge(vertices, normal, i, j, n, +1, -1);
            if (i+1 < n && j+1 < n)
                wedge(vertices, normal, i, j, n, +1, +1);

            // normalize normals
            normalize(normal);

            // store result
            normals[i*n+j] = normal;
        }
}

void display() {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRotated((frames++)*0.2, 0, 0, 1);

    if (frames % 500 == 0) {
        std::cout << frames*1000.0/glutGet(GLUT_ELAPSED_TIME) << std::endl;
    }

    glColor3d(0, 0, 1);
    glutSolidSphere(0.97, 100, 100);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_INDEX_ARRAY);


    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    #ifdef GPU
    propagate_gpu(N);
    validate_gpu(N);
    #else
    propagate((float3 *) vertices.data(), (float3*) velocities.data(), N);
    validate((float3 *) vertices.data(), N);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(),
                vertices.data(), GL_STREAM_DRAW);
    #endif
    glVertexPointer(3, GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glColorPointer(4, GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    #ifdef GPU
    update_normals_gpu(N);
    #else
    update_normals((float3*) vertices.data(), (float3*) normals.data(), N);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*normals.size(),
                 normals.data(), GL_STREAM_DRAW);
    #endif
    glNormalPointer(GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glDrawElements(GL_TRIANGLE_STRIP, indices.size(), GL_UNSIGNED_INT, NULL);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);

    if ((int)(frames*G) % 1000 == 0)
        init_data(N);

    glutSwapBuffers();
    glutPostRedisplay();
}

void init_data(int n) {

    // vertices
    vertices.resize(0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float x = i*4.0/(n-1)-2;
            float y = j*4.0/(n-1)-2;
            float z = 2;

            vertices.push_back(x); // x
            vertices.push_back(y); // y
            vertices.push_back(z); // z
        }
    }

    // device temp
    cudaMalloc(&Temp, sizeof(float3)*n*n);                               CUERR
    cudaMemset(Temp, 0, sizeof(float3)*n*n);                             CUERR

    // velocities
    velocities.resize(3*n*n);
    for (auto& v : velocities) v = 0;

    cudaMalloc(&Velocities, sizeof(float3)*n*n);                         CUERR
    cudaMemset(Velocities, 0, sizeof(float3)*n*n);                       CUERR

    // set constraints
    cnstr_two = vertices[3*n]-vertices[0];
    cnstr_dia = sqrt(2*cnstr_two*cnstr_two);

    // normals
    normals.resize(3*n*n);
    update_normals((float3*) vertices.data(), (float3*) normals.data(), n);

    // colors
    colors.resize(0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            colors.push_back(1);             // r
            colors.push_back(0);             // g
            colors.push_back(0);             // b
            colors.push_back(0.9);           // a
        }

    // indices
    indices.resize(0);
    for (int i = 0; i < n-1; i++) {
        int base = i*n;
        indices.push_back(base);
        for (int j = 0; j < n; j++) {
            indices.push_back(base+j);
            indices.push_back(base+j+n);
        }
        indices.push_back(base+2*n-1);
    }

    glGenBuffers(1, &vbo);
    glGenBuffers(1, &cbo);
    glGenBuffers(1, &nbo);
    glGenBuffers(1, &ibo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*vertices.size(),
                 vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);


    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*normals.size(),
                 normals.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&vbo_resource2, nbo, cudaGraphicsMapFlagsWriteDiscard);

    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*colors.size(),
                 colors.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*indices.size(),
                 indices.data(), GL_STATIC_DRAW);
}

void init_GL(int *argc, char **argv) {

    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Position-Based Dynamics");
    glutDisplayFunc(display);

    glewInit();

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLfloat mat_specular[] = { 0.8, 0.8, 0.8, 1.0 };
    GLfloat mat_shininess[] = { 50.0 };
    GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
    glShadeModel (GL_SMOOTH);

    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glViewport(0, 0, window_width, window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
    glTranslatef(0.0, 0.0, -6.0);
    glRotated(300, 1, 0, 0);
    glRotated(270, 0, 0, 1);
}

int main(int argc, char **argv) {
    init_GL(&argc, argv);
    init_data(N);
    glutMainLoop();
}
