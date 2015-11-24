package org.deeplearning4j.nnpractice;

import java.util.Random;

/**
 * Created by b1012059 on 2015/11/23.
 */
public class LogisticRegression {
    public int nIn;
    public int nOut;
    public double wIO[][];
    public double bias[];
    public int N;
    public Random rng;

    public LogisticRegression(int nIn, int nOut, int N){
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;

        wIO = new double[nOut][nIn];
        bias = new double[nOut];
    }

    public double[] train(double input[], int teach[], double learningLate){
        double nOutput[] = new double[nOut];
        return nOutput;
    }

    private static void testLogisticRegression(){
        int nInput = 5;
        int nOutput = 2;
        int epoch = 200;
        double learningLate = 0.1;

        double inputData[][] = {
                {1., 1., 1., 0., 0., 0.},
                {1., 0., 1., 0., 0., 0.},
                {1., 1., 1., 0., 0., 0.},
                {0., 0., 1., 1., 1., 0.},
                {0., 0., 1., 1., 0., 0.},
                {0., 0., 1., 1., 1., 0.}
        };

        int teachData[][] = {
                {1, 0},
                {1, 0},
                {1, 0},
                {0, 1},
                {0, 1},
                {0, 1}
        };

        LogisticRegression logReg = new LogisticRegression(nInput, nOutput, inputData.length);

        for(int count = 0; count < epoch; count++){
            for(int i = 0; i < inputData.length; i++){
                logReg.train(inputData[i], teachData[i], learningLate);
            }
        }
    }

    public static void main(String args[]){
        testLogisticRegression();
    }
}
