package org.deeplearning4j.nnpractice;

import java.util.Random;

/**
 * Created by b1012059 on 2015/09/08.
 */
public class utils {

    //シグモイド関数の傾き
    private final static double beta = 1.0;

    /**
     * 二項分布
     * @param n
     * @param p
     * @param rng
     * @return
     */
    public static int binomial(int n, double p, Random rng) {
        if(p < 0 || p > 1) return 0;

        int c = 0;
        double r;

        for(int i = 0; i < n; i++) {
            r = rng.nextDouble();
            if (r < p) c++;
        }

        return c;
    }

    /**
     *シグモイド関数
     * @param tmpOutput
     * @return sigmoid
     */
    public static double funSigmoid(double tmpOutput){
        double sigmoid = 1.0 / (1.0 + Math.exp(-beta * tmpOutput));
        return sigmoid;
    }

    /**
     * シグモイド関数の微分
     * @param tmpOutput
     * @return dsigmoid
     */
    public static double dfunSigmoid(double tmpOutput){
        double dsigmoid = tmpOutput * (1 - tmpOutput);
        return dsigmoid;
    }

    /**
     * ソフトマックス関数
     * @param tmpOutput
     * @return
     */
    public static void funSoftmax(double tmpOut[], int nOut){
        double max = 0.0;
        double sum = 0.0;

        for(int i = 0; i < nOut; i++){
            if(max < tmpOut[i]) max = tmpOut[i];
        }

        for(int i = 0; i < nOut; i++){
            tmpOut[i] = Math.exp(tmpOut[i] - max);
            sum += tmpOut[i];
        }

        for(int i = 0; i < nOut; i++) tmpOut[i] /= sum;
    }
}
