package org.deeplearning4j.nnpractice;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static org.deeplearning4j.nnpractice.utils.uniform;

/**
 * Created by b1012059 on 2016/01/31.
 */
public class ProjectionLayer {
    private static Logger log = LoggerFactory.getLogger(ProjectionLayer.class);
    private int N;
    private int vocab;
    private int dim;
    private double wDI[][];
    private Random rng;

    public ProjectionLayer(int N, int vocab, int dim, double wDI[][], Random rng){
        this.N = N;
        this.vocab = vocab;
        this.dim = dim;
        log.info("Initialize ProjectionLayer");

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng  = rng;

        if(wDI == null){
            this.wDI = new double[this.dim][this.vocab];
            //double element = 1.0 / nIn;
            for (int i = 0; i < this.dim; i++) {
                for (int j = 0; j < this.vocab; j++) {
                    this.wDI[i][j] = uniform(vocab, dim, rng, null);
                }
            }
        } else {
            this.wDI = wDI;
        }
    }

    public double[] lookUpTable(int wordToId){
        double[] lookUp = new double[dim];

        for(int j = 0; j < dim; j++){
            lookUp[j] = wDI[j][wordToId];
        }

        return lookUp;
    }

    public double[][] getwDI(){
        return  this.wDI;
    }

    public void backwardCal(int wordToId, double dProjection[]){

        for(int j = 0; j < dim; j++){
            wDI[j][wordToId] += dProjection[j];
        }
    }
}
