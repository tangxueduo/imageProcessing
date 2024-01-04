    # include <stdio.h>
    # include <math.h>

    int maxF(int x, int y){
        if (x>y){
            return x;
        }else
        {
            return y;
        }
    }

    int minF(int x, int y){
        if (x<y){
            return x;
        }else
        {
            return y;
        }
    }

    // height, width, ys, m, u, hu_volume_lpi, ratio=2.0
    void findCpr(short *im, short *cprIm, float *line, float *m, float *u, int height, int width, int imD, int imH, int imW)
    {
        float ratio=2.0;
        int wh = imH*imW;
        float a0=0,a1=0,a2=0,a3=1;
        float x=0,y=0,z=0;
        int x1, y1, x2, y2, fz, cz;
        float divV, div11, div12, div21, div22;
        float f11,f12,f21,f22,hu1,hu2;
        int i=0,j=0;
        for(j=0; j<height; j++){
            a0=line[j*3+0];
            a1=line[j*3+1];
            a2=line[j*3+2];
            for(i=0; i<width; i++){
                x=m[0] * a0 + m[4] * a1 + m[8] * a2 + m[12] * a3;
                y=m[1] * a0 + m[5] * a1 + m[9] * a2 + m[13] * a3;
                z=m[2] * a0 + m[6] * a1 + m[10] * a2 + m[14] * a3;
                if (x>=0 && x<imW*ratio && y>=0 && y<imH*ratio && z>=0 && z<imD*ratio){
                    x /= ratio;
                    y /= ratio;
                    z /= ratio;
                    if (z<0.0){
                        z=0.0;
                    };
                    if (z>(float)(imD-1)){
                        z=(float)(imD-1);
                    };

                    x1 = minF(maxF((int)x, 0), imW - 1);
                    y2 = minF(maxF((int)y, 0), imH - 1);
                    fz = minF(maxF((int)z, 0), imD - 1);
                    x2 = minF(maxF(x1 + 1, 0), imW - 1);
                    y1 = minF(maxF(y2 + 1, 0), imH - 1);
                    cz = minF(maxF(fz + 1, 0), imD - 1);

                    if (x1!=x2 && y1!=y2){
                        divV = (x2 - x1) * (y2 - y1);
                        div11 = ((float)x2 - x) * ((float)y2 - y) / divV;
                        div21 = (x - (float)x1) * ((float)y2 - y) / divV;
                        div12 = ((float)x2 - x) * (y - (float)y1) / divV;
                        div22 = (x - (float)x1) * (y - (float)y1) / divV;
                        
                        int id11 = (int)(fz*wh+y1*imW+x1);
                        int id21 = (int)(fz*wh+y1*imW+x2);
                        int id12 = (int)(fz*wh+y2*imW+x1);
                        int id22 = (int)(fz*wh+y2*imW+x2);
                        f11 = (float)im[id11];
                        f12 = (float)im[id12];
                        f21 = (float)im[id21];
                        f22 = (float)im[id22];
                        hu1 = f11*div11 + f21*div21 + f12*div12 + f22*div22;

                        id11 = (int)(cz*wh+y1*imW+x1);
                        id21 = (int)(cz*wh+y1*imW+x2);
                        id12 = (int)(cz*wh+y2*imW+x1);
                        id22 = (int)(cz*wh+y2*imW+x2);
                        f11 = (float)im[id11];
                        f12 = (float)im[id12];
                        f21 = (float)im[id21];
                        f22 = (float)im[id22];
                        hu2 = f11*div11 + f21*div21 + f12*div12 + f22*div22;

                        cprIm[j*width+i] = (short)(((float)cz - z) * hu1 + (z - (float)fz) * hu2);
                    }
                }
                a0 += u[0];
                a1 += u[1];
                a2 += u[2];
            }
        }
    }

    void findLumen(short *im, short *lumenIm, float *points, float *normals, float *tangents, float *m, float averageDistance, float radTheta,  int height, int width, int imD, int imH, int imW)
    {
        float ratio=2.0;
        int wh = imH*imW;
        float a0=0,a1=0,a2=0,a3=1;
        float n0,n1,n2;
        float x=0,y=0,z=0;
        int x1, y1, x2, y2, fz, cz;
        float divV, div11, div12, div21, div22;
        float f11,f12,f21,f22,hu1,hu2;
        int i=0,j=0;
        float halfHeight = (float)(height/2);
        for(i=0; i<width; i++){
            a0=points[i*3+0];
            a1=points[i*3+1];
            a2=points[i*3+2];
            n0=normals[i*3+0]*averageDistance;
            n1=normals[i*3+1]*averageDistance;
            n2=normals[i*3+2]*averageDistance;

            float t0 = tangents[i*3+0];
            float t1 = tangents[i*3+1];
            float t2 = tangents[i*3+2];
            float length = (float)sqrt((double)(t0 * t0 + t1 * t1 + t2 * t2));
            if (length>1e-10) {
                length = 1.0/length;
                t0 *= length;
                t1 *= length;
                t2 *= length;

                float s = (float)sin(radTheta);
                float c = (float)cos(radTheta);
                float t = 1-c;

                float r[16] = {
                    t0*t0*t+c, t1*t0*t+t2*s, t2*t0*t-t1*s, 0.0,
                    t0*t1*t-t2*s, t1*t1*t+c, t2*t1*t+t0*s,0.0,
                    t0*t2*t+t1*s, t1*t2*t-t0*s, t2*t2*t+c, 0.0,
                    0.0,0.0,0.0,1.0};

                float w = r[3]*n0 + r[7]*n1 + r[11]*n2 + r[15];
                if (w==0){
                    w=1.0;
                };
                float n[3] = {
                    (r[0]*n0 + r[4]*n1 + r[8]*n2 + r[12])/w,
                    (r[1]*n0 + r[5]*n1 + r[9]*n2 + r[13])/w,
                    (r[2]*n0 + r[6]*n1 + r[10]*n2 + r[14])/w,
                };

                float scaleN[4] = {n[0]*halfHeight,n[1]*halfHeight,n[2]*halfHeight, 0.0};
                a0 -= scaleN[0];
                a1 -= scaleN[1];
                a2 -= scaleN[2];
                a3 -= scaleN[3];

                for(j=0; j<height; j++){
                    x=m[0] * a0 + m[4] * a1 + m[8] * a2 + m[12] * a3;
                    y=m[1] * a0 + m[5] * a1 + m[9] * a2 + m[13] * a3;
                    z=m[2] * a0 + m[6] * a1 + m[10] * a2 + m[14] * a3;
                    if (x>=0 && x<imW*ratio && y>=0 && y<imH*ratio && z>=0 && z<imD*ratio){
                        x /= ratio;
                        y /= ratio;
                        z /= ratio;
                        if (z<0.0){
                            z=0.0;
                        };
                        if (z>(float)(imD-1)){
                            z=(float)(imD-1);
                        };

                        x1 = minF(maxF((int)x, 0), imW - 1);
                        y2 = minF(maxF((int)y, 0), imH - 1);
                        fz = minF(maxF((int)z, 0), imD - 1);
                        x2 = minF(maxF(x1 + 1, 0), imW - 1);
                        y1 = minF(maxF(y2 + 1, 0), imH - 1);
                        cz = minF(maxF(fz + 1, 0), imD - 1);

                        if (x1!=x2 && y1!=y2){
                            divV = (x2 - x1) * (y2 - y1);
                            div11 = ((float)x2 - x) * ((float)y2 - y) / divV;
                            div21 = (x - (float)x1) * ((float)y2 - y) / divV;
                            div12 = ((float)x2 - x) * (y - (float)y1) / divV;
                            div22 = (x - (float)x1) * (y - (float)y1) / divV;
                            
                            int id11 = (int)(fz*wh+y1*imW+x1);
                            int id21 = (int)(fz*wh+y1*imW+x2);
                            int id12 = (int)(fz*wh+y2*imW+x1);
                            int id22 = (int)(fz*wh+y2*imW+x2);
                            f11 = (float)im[id11];
                            f12 = (float)im[id12];
                            f21 = (float)im[id21];
                            f22 = (float)im[id22];
                            hu1 = f11*div11 + f21*div21 + f12*div12 + f22*div22;

                            id11 = (int)(cz*wh+y1*imW+x1);
                            id21 = (int)(cz*wh+y1*imW+x2);
                            id12 = (int)(cz*wh+y2*imW+x1);
                            id22 = (int)(cz*wh+y2*imW+x2);
                            f11 = (float)im[id11];
                            f12 = (float)im[id12];
                            f21 = (float)im[id21];
                            f22 = (float)im[id22];
                            hu2 = f11*div11 + f21*div21 + f12*div12 + f22*div22;
                            short hu=(short)(((float)cz - z) * hu1 + (z - (float)fz) * hu2);
                            lumenIm[j*width+i] = hu;
                        }
                    }
                    a0 += n[0];
                    a1 += n[1];
                    a2 += n[2];
                }
            }
        }
    }

    // gcc -shared -O2 forCpr.c  -ldl -o forCpr.so