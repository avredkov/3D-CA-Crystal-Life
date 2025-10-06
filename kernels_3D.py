ker_code="""
#include <math.h>                   
#include <curand.h>
#include <curand_kernel.h>


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"{

#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )
#define _Z  ( threadIdx.z + blockIdx.z * blockDim.z )

#define _SIZE_X ( blockDim.x * gridDim.x )
#define _SIZE_Y ( blockDim.y * gridDim.y  )
#define _SIZE_Z ( blockDim.z * gridDim.z  )
#define _DIF_SIZE_X  ( 4*blockDim.x * gridDim.x )
#define _DIF_SIZE_Y  ( 4*blockDim.y * gridDim.y )
#define _DIF_SIZE_Z  ( 4*blockDim.z * gridDim.z )

#define MAX 4
#define _XM(x)  ( (x + _SIZE_X) % _SIZE_X )
#define _YM(y)  ( (y + _SIZE_Y) % _SIZE_Y )
#define _ZM(z)  ( (z + _SIZE_Z) % _SIZE_Z )
#define _INDEX(x,y,z)  (_XM(x)* _SIZE_Z* _SIZE_Y + _YM(y)* _SIZE_Z +_ZM(z) )

#define _DIF_XM(x)  ( (x + _DIF_SIZE_X) % _DIF_SIZE_X )
#define _DIF_YM(y)  ( (y + _DIF_SIZE_Y) % _DIF_SIZE_Y )
#define _DIF_ZM(z)  ( (z + _DIF_SIZE_Z) % _DIF_SIZE_Z )
#define _DIF_INDEX(x,y,z)  ( _DIF_XM(x)*_DIF_SIZE_Y*_DIF_SIZE_Z  + _DIF_YM(y) * _DIF_SIZE_Z+ _DIF_ZM(z))




/*__device__ int nbrs(int x, int y, int * in)
{
     return ( int(in[ _INDEX(x, y+1) ]>in[ _INDEX(x, y)]) \
                   + int(in[ _INDEX(x, y-1) ]>in[ _INDEX(x, y)]) \
                   + int(in[ _INDEX(x+1, y) ]>in[ _INDEX(x, y)]) \
                   + int(in[ _INDEX(x-1, y) ]>in[ _INDEX(x, y)]) );
}*/





    
__device__ int read_atom_with_walls(const int *arr, int x, int y, int z, int wx0, int wx1, int wy0, int wy1, int wz0, int wz1)
{
    if (x < 0) { if (wx0) return 0; else x = _XM(x); } else if (x >= _SIZE_X) { if (wx1) return 0; else x = _XM(x); }
    if (y < 0) { if (wy0) return 0; else y = _YM(y); } else if (y >= _SIZE_Y) { if (wy1) return 0; else y = _YM(y); }
    if (z < 0) { if (wz0) return 0; else z = _ZM(z); } else if (z >= _SIZE_Z) { if (wz1) return 0; else z = _ZM(z); }
    return arr[_INDEX(x,y,z)];
}

__global__ void diffuse_adatoms(int * atoms,  float * directions, int cycleX, int cycleY,int cycleZ, float biasX, float biasY, float biasZ, int wx0, int wx1, int wy0, int wy1, int wz0, int wz1)
    { 
    
             int x = 4*_X+cycleX;
             int y = 4*_Y+cycleY;
             int z = 4*_Z+cycleZ; 
             int dx=0;
             int dy=0;
             int dz=0;
             int hop=0;
               
             if (atoms[_DIF_INDEX(x,y,z)] == 1)
             {
               
                if (directions[_DIF_INDEX(x,y,z)]<0.33333333)
                {
                    if (directions[_DIF_INDEX(x,y,z)]<0.16666666)
                    {
                        dx=1;
                    }
                    if (directions[_DIF_INDEX(x,y,z)]>=0.16666666)
                    {
                        dx=-1;
                    }
                } 
                if (directions[_DIF_INDEX(x,y,z)]>=0.33333333 && directions[_DIF_INDEX(x,y,z)]<0.66666666)
                {
                    if (directions[_DIF_INDEX(x,y,z)]<0.5)
                    { 
                        dy=1;
                    }
                    if (directions[_DIF_INDEX(x,y,z)]>=0.5)
                    {
                         dy=-1;
                    }     
                }
                if (directions[_DIF_INDEX(x,y,z)]>=0.66666666)
                {
                    if (directions[_DIF_INDEX(x,y,z)]<0.83333333)
                    { 
                        dz=1;
                    }
                    if (directions[_DIF_INDEX(x,y,z)]>=0.83333333)
                    {
                         dz=-1;
                    }     
                }
                

                int blocked = 0;
                if (dx == -1 && x == 0 && wx0) blocked = 1;
                if (dx ==  1 && x == _DIF_SIZE_X - 1 && wx1) blocked = 1;
                if (dy == -1 && y == 0 && wy0) blocked = 1;
                if (dy ==  1 && y == _DIF_SIZE_Y - 1 && wy1) blocked = 1;
                if (dz == -1 && z == 0 && wz0) blocked = 1;
                if (dz ==  1 && z == _DIF_SIZE_Z - 1 && wz1) blocked = 1;

                if (!blocked && atoms[_DIF_INDEX(x+dx,y+dy,z+dz)] == 0 ) 
                {
                    hop=1; 
                }    
            }
                
           __syncthreads();  
          atoms[_DIF_INDEX(x,y,z)] -=hop; 
          atoms[_DIF_INDEX(x+dx,y+dy,z+dz)] +=hop; 
          __syncthreads();
        
                
    }
    

__global__ void calculate_coordinations(int * atoms, int * coordination_count, int * coordination, int wx0, int wx1, int wy0, int wy1, int wz0, int wz1)
    { 
    int x = _X, y = _Y, z= _Z; 
    int neighbours=0;
    if(atoms[_INDEX(x,y,z)]==2)
    {
    neighbours+=int(read_atom_with_walls(atoms,x-1,y,z,wx0,wx1,wy0,wy1,wz0,wz1)==2)+int(read_atom_with_walls(atoms,x+1,y,z,wx0,wx1,wy0,wy1,wz0,wz1)==2)+int(read_atom_with_walls(atoms,x,y-1,z,wx0,wx1,wy0,wy1,wz0,wz1)==2)+int(read_atom_with_walls(atoms,x,y+1,z,wx0,wx1,wy0,wy1,wz0,wz1)==2)+int(read_atom_with_walls(atoms,x,y,z-1,wx0,wx1,wy0,wy1,wz0,wz1)==2)+int(read_atom_with_walls(atoms,x,y,z+1,wx0,wx1,wy0,wy1,wz0,wz1)==2);
          
    atomicAdd(&coordination_count[neighbours],1);  
    coordination[_INDEX(x,y,z)]=neighbours;
    }
    else
    {
     coordination[_INDEX(x,y,z)]=0;
     }
    __syncthreads();  
     

     }
    

__device__ bool are_face_sharing_neighbors(int dx1, int dy1, int dz1, int dx2, int dy2, int dz2) {
    // Two neighbors are adjacent only if they share a face with each other
    // This means they must differ by 1 in the same dimension
    if (dx1 != 0 || dy1 != 0 || dz1 != 0) {  // first direction is valid
        if (dx2 != 0 || dy2 != 0 || dz2 != 0) {  // second direction is valid
            // Check if they share a vertex with the central cell and with each other
            // For face-sharing neighbors, only one coordinate should differ between them
            return ((dx1 != 0 && dy2 != 0 && abs(dx1) + abs(dy2) == 2) ||  // x-y plane
                   (dx1 != 0 && dz2 != 0 && abs(dx1) + abs(dz2) == 2) ||  // x-z plane
                   (dy1 != 0 && dz2 != 0 && abs(dy1) + abs(dz2) == 2) ||  // y-z plane
                   (dy1 != 0 && dx2 != 0 && abs(dy1) + abs(dx2) == 2) ||  // y-x plane
                   (dz1 != 0 && dx2 != 0 && abs(dz1) + abs(dx2) == 2) ||  // z-x plane
                   (dz1 != 0 && dy2 != 0 && abs(dz1) + abs(dy2) == 2));   // z-y plane
        }
    }
    return false;
}

__device__ bool are_three_face_sharing(int dx1, int dy1, int dz1, 
                                     int dx2, int dy2, int dz2,
                                     int dx3, int dy3, int dz3) {
    // Three neighbors must form a connected chain through face-sharing
    return (are_face_sharing_neighbors(dx1, dy1, dz1, dx2, dy2, dz2) &&
            are_face_sharing_neighbors(dx2, dy2, dz2, dx3, dy3, dz3) &&
            are_face_sharing_neighbors(dx1, dy1, dz1, dx3, dy3, dz3));
}

__global__ void analyze_neighbors(const int* input, int* results, int * states, int wx0, int wx1, int wy0, int wy1, int wz0, int wz1, const int* prev_states, unsigned int* site_age, int enable_age, int do_count)
{
    int x = _X;
    int y = _Y;
    int z = _Z;
    int idx = _INDEX(x, y, z);
    // default state
    if (x >= _SIZE_X || y >= _SIZE_Y || z >= _SIZE_Z) return;
    states[idx] = 0;
    if(input[idx]!=2)
    {
        
        
        // Define the six nearest neighbors (face-sharing only)
        const int directions[6][3] = {
            {1, 0, 0},   // right  (+x)
            {-1, 0, 0},  // left   (-x)
            {0, 1, 0},   // up     (+y)
            {0, -1, 0},  // down   (-y)
            {0, 0, 1},   // front  (+z)
            {0, 0, -1}   // back   (-z)
        };
        
        // Count neighbors with value 2
        int count = 0;
        int neighbor_positions[6][3] = {0};  // Store positions of neighbors with value 2
        int pos_count = 0;
        
        for (int i = 0; i < 6; i++)
        {
            int nx = x + directions[i][0];
            int ny = y + directions[i][1];
            int nz = z + directions[i][2];
            
            if (read_atom_with_walls(input, nx, ny, nz, wx0, wx1, wy0, wy1, wz0, wz1) == 2) 
            {
                neighbor_positions[pos_count][0] = directions[i][0];
                neighbor_positions[pos_count][1] = directions[i][1];
                neighbor_positions[pos_count][2] = directions[i][2];
                pos_count++;
                count++;
            }
        }
        
        // Case 1: Exactly one neighbor equals 2
        if (do_count) atomicAdd(&results[0], (count == 1) ? 1 : 0);
        if (count==1)
        {
         states[idx]=1;
        }
        // Case 2: Exactly two face-sharing neighbors equal 2
        if (count == 2) {
            bool adjacent = are_face_sharing_neighbors(
                neighbor_positions[0][0], neighbor_positions[0][1], neighbor_positions[0][2],
                neighbor_positions[1][0], neighbor_positions[1][1], neighbor_positions[1][2]
            );
            if (adjacent==1)
            {
                states[_INDEX(x, y, z)]=2;
            }
            if (do_count) atomicAdd(&results[1], adjacent ? 1 : 0);
        }
        
        // Case 3: Exactly three face-sharing neighbors in a chain equal 2
        if (count == 3) {
            bool triple_adjacent = are_three_face_sharing(
                neighbor_positions[0][0], neighbor_positions[0][1], neighbor_positions[0][2],
                neighbor_positions[1][0], neighbor_positions[1][1], neighbor_positions[1][2],
                neighbor_positions[2][0], neighbor_positions[2][1], neighbor_positions[2][2]
            );
            if (triple_adjacent==1)
            {
                states[_INDEX(x, y, z)]=3;
            }
            if (do_count) atomicAdd(&results[2], triple_adjacent ? 1 : 0);
        }
    }
    // Update site ages if enabled
    if (enable_age)
    {
        unsigned int prev = (prev_states ? (unsigned int)prev_states[idx] : 0u);
        unsigned int now = (unsigned int)states[idx];
        if (now == 0u)
        {
            site_age[idx] = 0u;
        }
        else
        {
            if (now == prev && prev != 0u)
            {
                unsigned int a = site_age[idx];
                // saturate at max 32-bit
                site_age[idx] = (a == 0xFFFFFFFFu) ? a : (a + 1u);
            }
            else
            {
                site_age[idx] = 1u;
            }
        }
    }
}
    
__global__ void calculate_atoms_in_box(int * atoms, int *number, int box_size, int box_center_x, int box_center_y, int box_center_z)
    { 
    int x = _X, y = _Y, z= _Z; 
    if((x>box_center_x-box_size && x<=box_center_x+box_size) && (y>box_center_y-box_size && y<=box_center_x+box_size) && (y>box_center_y-box_size && y<=box_center_x+box_size))          
    if(atoms[_INDEX(x,y,z)]==2)
    {
    atomicAdd(&number[0],1);  
    }
    __syncthreads();  
     

     }   

__global__ void calculate_atoms_in_box_and_Rmax(int * atoms, int *stats,  int box_center_x, int box_center_y, int box_center_z)
    { 
    int x = _X, y = _Y, z= _Z; 
    float old=0.0;
    float distance_to_center=0.0;
    __syncthreads();  
    if(atoms[_INDEX(x,y,z)]==2)
    {
    atomicAdd(&stats[0],1); 
    distance_to_center=((x-box_center_x)^2+(y-box_center_y)^2+(z-box_center_z)^2);
    __syncthreads();  
      old=atomicMax(&stats[1],int(distance_to_center)); 
    
    }
    __syncthreads();  
     

     }   
    
        
    


__global__ void conway_ker(const int * atoms_in, int * atoms_out, const float * probability, const float* Ru, int wx0, int wx1, int wy0, int wy1, int wz0, int wz1, unsigned int* cryst_age, int enable_age)
    {

    int x = _X, y = _Y, z=_Z; 
    int to_encorporate=0;
    int idx = _INDEX(x,y,z);
    int current_state = atoms_in[idx];
    float local_probability = probability[idx];

      
     
    //if(x>15)
    {
        if(current_state==1)
        {
        
         //# 1st position - self, 2nd - right, 3rd down, 4th - left 5 - up, 
       
           int right  = read_atom_with_walls(atoms_in, x+1, y, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int down   = read_atom_with_walls(atoms_in, x, y+1, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int left   = read_atom_with_walls(atoms_in, x-1, y, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int up     = read_atom_with_walls(atoms_in, x, y-1, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int back   = read_atom_with_walls(atoms_in, x, y, z-1, wx0, wx1, wy0, wy1, wz0, wz1);
           int front  = read_atom_with_walls(atoms_in, x, y, z+1, wx0, wx1, wy0, wy1, wz0, wz1);
           int index=729*atoms_in[idx]+243*right+81*down+27*left+9*up+3*back+front;
        
            if (index>=0 && index<2187)
            {
                if(local_probability<Ru[index])
                {
                    to_encorporate=1;
                } 
    
           
            } 
              
        }
        if(current_state==2)
        {
        
         //# 1st position - self, 2nd - right, 3rd down, 4th - left 5 - up, 
       
           int right2  = read_atom_with_walls(atoms_in, x+1, y, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int down2   = read_atom_with_walls(atoms_in, x, y+1, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int left2   = read_atom_with_walls(atoms_in, x-1, y, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int up2     = read_atom_with_walls(atoms_in, x, y-1, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int back2   = read_atom_with_walls(atoms_in, x, y, z-1, wx0, wx1, wy0, wy1, wz0, wz1);
           int front2  = read_atom_with_walls(atoms_in, x, y, z+1, wx0, wx1, wy0, wy1, wz0, wz1);
           int index=729*atoms_in[idx]+243*right2+81*down2+27*left2+9*up2+3*back2+front2;
        
         
            {
                if(index>=0 && index<2187)
                {
                    if(local_probability<Ru[index])
                    {
                        to_encorporate=-1;
                    }
                } 
    
           
            } 
              
        }
       
    }
   
    int new_state = current_state + to_encorporate;
    atoms_out[idx] = new_state;
    if (enable_age)
    {
        if (new_state == 2)
        {
            if (current_state != 2)
            {
                cryst_age[idx] = 1u;
            }
            else
            {
                unsigned int a = cryst_age[idx];
                cryst_age[idx] = (a == 0xFFFFFFFFu) ? a : (a + 1u);
            }
        }
        else
        {
            cryst_age[idx] = 0u;
        }
    }
    
    }
    

__global__ void conway_ker_event_calc(const int * atoms_in, int * atoms_out, const float * probability, const float* Ru, int *events, int wx0, int wx1, int wy0, int wy1, int wz0, int wz1, unsigned int* cryst_age, int enable_age)
    {

    int x = _X, y = _Y, z=_Z; 
    int to_encorporate=0;
    int idx = _INDEX(x,y,z);
    int current_state = atoms_in[idx];
    float local_probability = probability[idx];

      
     
    //if(z<5 && x>0 && x<_SIZE_X-1 &&  y>0 && y<_SIZE_Y-1) // no nucleation at borders
    {
        if(current_state==1 )
        {
        
         //# 1st position - self, 2nd - right, 3rd down, 4th - left 5 - up, 
       
           int right3  = read_atom_with_walls(atoms_in, x+1, y, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int down3   = read_atom_with_walls(atoms_in, x, y+1, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int left3   = read_atom_with_walls(atoms_in, x-1, y, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int up3     = read_atom_with_walls(atoms_in, x, y-1, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int back3   = read_atom_with_walls(atoms_in, x, y, z-1, wx0, wx1, wy0, wy1, wz0, wz1);
           int front3  = read_atom_with_walls(atoms_in, x, y, z+1, wx0, wx1, wy0, wy1, wz0, wz1);
           int index=729*atoms_in[idx]+243*right3+81*down3+27*left3+9*up3+3*back3+front3;
        
            if (index>=0 && index<2187)
            {
                if(local_probability<Ru[index])
                {
                    to_encorporate=1;
                    atomicAdd(&events[index], int(1));
                } 
    
           
            } 
              
        }
        if(current_state==2)
        {
        
         //# 1st position - self, 2nd - right, 3rd down, 4th - left 5 - up, 
       
           int right4  = read_atom_with_walls(atoms_in, x+1, y, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int down4   = read_atom_with_walls(atoms_in, x, y+1, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int left4   = read_atom_with_walls(atoms_in, x-1, y, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int up4     = read_atom_with_walls(atoms_in, x, y-1, z, wx0, wx1, wy0, wy1, wz0, wz1);
           int back4   = read_atom_with_walls(atoms_in, x, y, z-1, wx0, wx1, wy0, wy1, wz0, wz1);
           int front4  = read_atom_with_walls(atoms_in, x, y, z+1, wx0, wx1, wy0, wy1, wz0, wz1);
           int index=729*atoms_in[idx]+243*right4+81*down4+27*left4+9*up4+3*back4+front4;
        
         
            {
                if(index>=0 && index<2187)
                {
                    if(local_probability<Ru[index])
                    {
                        to_encorporate=-1;
                        atomicAdd(&events[index], int(1));
                    } 
                }
    
           
            } 
              
        }
       
    }
   
    int new_state = current_state + to_encorporate;
    atoms_out[idx] = new_state;
    if (enable_age)
    {
        if (new_state == 2)
        {
            if (current_state != 2)
            {
                cryst_age[idx] = 1u;
            }
            else
            {
                unsigned int a = cryst_age[idx];
                cryst_age[idx] = (a == 0xFFFFFFFFu) ? a : (a + 1u);
            }
        }
        else
        {
            cryst_age[idx] = 0u;
        }
    }
           
    }    
    

    


}     
                                         
 
"""