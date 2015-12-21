package org.deeplearning4j.nnpractice;

import java.util.Random;
import static org.deeplearning4j.nnpractice.utils.*;

/**
 * Created by b1012059 on 2015/11/22.
 */
public class SimpleHiddenLayer extends HiddenLayer{
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
    public SimpleHiddenLayer(int nIn, int nOut, double wIO[][], double bias[], int N, Random rng){
        super(nIn, nOut, wIO, bias, N, rng, null);
        this.nIn = nIn;
        this.nOut = nOut;
        this.N = N;

        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        if(wIO == null){
            this.wIO = new double[nOut][nIn];
            for(int i = 0; i < nOut; i++){
                for(int j = 0; j < nIn; j++){
                    this.wIO[i][j] = uniform(nIn, nOut, rng, null);
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


    }

    /**
     * Hidden Layer Output
     * @param input
     * @param w
     * @param bias
     * @return
     */
    public double hOutput(double input[], double w[], double bias){
        double tmpData = 0.0;
        for(int i = 0; i < nIn; i++){
            tmpData += input[i] * w[i];
        }
        tmpData += bias;
        return funSigmoid(tmpData);
    }

    /**
     *
     * @param input
     * @param sample
     */
    public void sampleHgive(double input[], double sample[]){
        for(int i = 0; i < nOut; i++) sample[i] = binomial(1, hOutput(input, wIO[i], bias[i]), rng);
    }
}
