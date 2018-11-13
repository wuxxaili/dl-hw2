#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

float FindMax(image im, int size, int x, int y, int c){
    float max = im.data[x + im.w*(y+im.h*c)];
    int i,j;
    if(size%2 == 1){
        for(i = x-size/2; i <= x+size/2; ++i){
            for(j = y-size/2; j <= y+size/2; ++j){
                if( 0<= i&& i < im.w && 0<= j && j <im.h){
                    if(im.data[i + im.w*(j+im.h*c)] > max){
                        max = im.data[i + im.w*(j+im.h*c)];
                    }
                }
            }
        }
    }
    else{
        for(i = x; i < x+size; ++i){
            for(j = y; j < y+size; ++j){
                if( 0<= i&& i < im.w && 0<= j && j <im.h){
                    if(im.data[i + im.w*(j+im.h*c)] > max){
                        max = im.data[i + im.w*(j+im.h*c)];
                    }
                }
            }
        }
    }
    return max;
}

matrix maxpool(image im, int size,int stride)
{
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int cols = outw * outh * im.c;
    matrix col = make_matrix(1,cols);
    int j,c,x,y;
    for (c = 0; c < im.c; ++c){
        for (j = 0; j<outw*outh; ++j){
            x = stride*(j/outw);
            y = stride*(j%outw);
            col.data[j+c*outh*outh] = FindMax(im,size,x,y, c);
        }
    }
    return col;
}

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int i,j;
    for(i = 0;i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        matrix mp = maxpool(example, l.size, l.stride);
        for(j = 0; j < outw*outh*l.channels; ++j){
            out.data[i*out.cols + j] = mp.data[j];
        }
        free_matrix(mp);
    }
    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}
typedef struct{
    int num[2];
}position;

position FindMaxIndex(image im, int size,int x,int y,int c){
    position out;
    out.num[0] = x;
    out.num[1] = y;
    float max = im.data[x + im.w*(y+im.h*c)];
    int i,j;
    if(size%2 == 1){
        for(i = x-size/2; i <= x+size/2; ++i){
            for(j = y-size/2; j <= y+size/2; ++j){
                if( 0<= i&& i  < im.w && 0<= j && j <im.h){
                    if(im.data[i + im.w*(j+im.h*c)] > max){
                        max = im.data[i + im.w*(j+im.h*c)];
                        out.num[0] = i;
                        out.num[1] = j;
                    }}}}}
    else{
        for(i = x; i < x+size; ++i){
            for(j = y; j < y+size; ++j){
                if(0<= i&& i  < im.w && 0<= j && j <im.h){
                    if(im.data[i + im.w*(j+im.h*c)] > max){
                        max = im.data[i + im.w*(j+im.h*c)];
                        out.num[0] = i;
                        out.num[1] = j;
                    }}}}}
    return out;
}

void ChangePrev(image in, image prev,int size,int stride,matrix delta){
    int outw = (in.w-1)/stride + 1;
    int outh = (in.h-1)/stride + 1;
    int cols = outw * outh * in.c;
    //matrix col = make_matrix(1,cols);
    int i,j,c,x,y;
    for (c = 0; c < in.c; ++c){
        for (j = 0; j<outw*outh; ++j){
            x = stride*(j/outw);
            y = stride*(j%outw);
            position pos = FindMaxIndex(in,size,x,y, c);
            prev.data[pos.num[0] + prev.w*(pos.num[1]+prev.h*c)] += delta.data[c*outw*outh + j];
        }
    }
}
// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    
    int i;
    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        image dexample = float_to_image(prev_delta.data + i*in.cols, l.width, l.height, l.channels);
        assert(in.cols == l.width*l.height*l.channels);
        
        delta.rows = 1;
        delta.cols = outw*outh*l.channels;
        delta.data = l.delta[0].data + i*delta.rows*delta.cols;
        if(prev_delta.data){
            //matrix col = matmul(wt, delta);
            ChangePrev(example, dexample,l.size,l.stride,delta);
        }
    }
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

