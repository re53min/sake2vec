package org.deeplearning4j.nnpractice;

import java.util.Random;
import static org.deeplearning4j.nnpractice.utils.*;

/**
 * Created by b1012059 on 2015/09/09.
 */
public class HiddenLayer {
    public int nIn;
    public int nOut;
    public double wIO[][];
    public double bias[];
    public int N;
    public Random rng;


    /**
     *
     * @param nIn
     * @param nOut
     * @param wIO
     * @param bias
     * @param N
     * @param rng
     */
    public HiddenLayer(int nIn, int nOut, double wIO[][], double bias[], int N, Random rng, String activation){
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;

        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        if(wIO == null){
            this.wIO = new double[nOut][nIn];
            for(int i = 0; i < nOut; i++){
                for(int j = 0; j < nIn; j++){
                    this.wIO[i][j] = uniform(nIn, nOut, rng, activation);
                }
            }
        } else{
            this.wIO = wIO;
        }

        if(bias == null){
            this.bias = new double[nOut];
            for(int i = 0; i < nOut; i++){
                this.bias[i] = 0;
            }
        } else {
            this.bias = bias;
        }

        /*
        ここラムダ式で記述
         */

    }

    /**
     * Forward Calculation
     * @param input
     * @param output
     */
    public void forwardCal(double input[], double output[]){
        for(int i = 0; i < nOut; i++) {
            output[i] = this.output(input, wIO[i], bias[i]);
        }
    }

    /**
     * Hidden Layer Output
     * @param input
     * @param w
     * @param bias
     * @return
     */
    public double output(double input[], double w[], double bias){
        double tmpData = 0.0;
        for(int i = 0; i < nIn; i++){
            tmpData += input[i] * w[i];
        }
        tmpData += bias;

        return funSigmoid(tmpData);
    }

    /*
    ここ逆方向計算
     */
    public void backwardCal(double input[], double dOutput[], double prevInput[],
                            double prevdOutput[], double prevWIO[][], double learningLate){


    }
}
