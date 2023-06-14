    /*
    author: zjin 2021.08.21
    */
    # include <stdio.h>

    // bound = [zmin,zmax, ymin,ymax, xmin,xmax]
    void boundbox_uint8(unsigned char *im, int *bound, int depth, int height, int width)
    {
        int wh = height*width;
        int k,i,j;
        for(k=0; k<depth; k++){
            for(j=0; j<height; j++){
                for(i=0; i<width; i++){
                    int idx = (int)(k*wh+j*width+i);
                    if(im[idx]){
                       if(bound[4]>i){
                           bound[4]=i;
                       }else{
                            if (bound[5]<i){
                                bound[5]=i;
                            }
                       }

                       if(bound[2]>j){
                           bound[2]=j;
                       }else{
                            if (bound[3]<j){
                                bound[3]=j;
                            }
                       }

                       if(bound[0]>k){
                           bound[0]=k;
                       }else{
                            if (bound[1]<k){
                                bound[1]=k;
                            }
                       }
                    }
                }
            }
        }
    }

    // bound = [zmin,zmax, ymin,ymax, xmin,xmax]
    void boundbox_bool(_Bool *im, int *bound, int depth, int height, int width)
    {
        int wh = height*width;
        int k,i,j;
        for(k=0; k<depth; k++){
            for(j=0; j<height; j++){
                for(i=0; i<width; i++){
                    int idx = (int)(k*wh+j*width+i);
                    if(im[idx]){
                       if(bound[4]>i){
                           bound[4]=i;
                       }else{
                            if (bound[5]<i){
                                bound[5]=i;
                            }
                       }

                       if(bound[2]>j){
                           bound[2]=j;
                       }else{
                            if (bound[3]<j){
                                bound[3]=j;
                            }
                       }
                       
                       if(bound[0]>k){
                           bound[0]=k;
                       }else{
                            if (bound[1]<k){
                                bound[1]=k;
                            }
                       }
                    }
                }
            }
        }
    }


    // gcc -shared -O2 bound_box.c  -ldl -o bound_box.so