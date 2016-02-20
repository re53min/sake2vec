package org.deeplearning4j.nnpractice;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;
import java.util.function.DoubleFunction;

import static org.deeplearning4j.nnpractice.utils.*;

/**
 * Hidden Layer
 * Created by b1012059 on 2015/09/09.
 */
public class HiddenLayer {
    private static Logger log = LoggerFactory.getLogger(HiddenLayer.class);
    private int nIn;
    private int nOut;
    private double wIO[][];
    private double bias[];
    private int N;
    private Random rng;
    private DoubleFunction<Double> activation;
    private DoubleFunction<Double> dActivation;


    /**
     * Hidden Layer Constructor
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

        log.info("Initialize HiddenLayer");

        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //重み行列を連続一様分布で初期化
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

        //バイアスを0で初期化
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
        if (activation == "sigmoid" || activation == null) {
            this.activation = (double tmpOut) -> funSigmoid(tmpOut);
            this.dActivation = (double tmpOut) -> dfunSigmoid(tmpOut);
        } else if(activation == "tanh"){
            this.activation = (double tmpOut) -> funTanh(tmpOut);
            this.dActivation = (double tmpOut) -> dfunTanh(tmpOut);
        } else if(activation == "ReLU"){
            this.activation = (double tmpOut) -> funReLU(tmpOut);
            this.dActivation = (double tmpOut) -> dfunReLU(tmpOut);
        } else {
            log.info("Activation function not supported!");
        }

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
    private double output(double input[], double w[], double bias){
        double tmpData = 0.0;
        for(int i = 0; i < nIn; i++){
            tmpData += input[i] * w[i];
        }
        tmpData += bias;

        return activation.apply(tmpData);
    }

    /**
     * Backward Calculation
     * @param input 今層への入力(前層の出力)
     * @param dOutput 今層の誤差勾配
     * @param prevInput 次層への入力(今層の出力)
     * @param prevdOutput　次層の誤差勾配
     * @param prevWIO 次層の重み行列(今層→次層への重み行列)
     * @param learningRate 学習率
     */
    public void backwardCal(double input[], double dOutput[], double prevInput[],
                            double prevdOutput[], double prevWIO[][], double learningRate){

        if(dOutput == null) dOutput = new double[nOut];

        //今層の誤差勾配
        for(int i = 0; i < nOut; i++) {
            dOutput[i] = 0;
            for (int j = 0; j < prevdOutput.length; j++) {
                //次層の誤差勾配と次層への重み行列との積
                dOutput[i] += prevdOutput[j] * prevWIO[j][i];
            }
            //活性化関数の微分との積により誤差勾配(修正量)を求める
            dOutput[i] *= dActivation.apply(prevInput[i]);
        }

        //今層の誤差勾配を用いて重み行列及びバイアスの更新
        for(int i = 0; i < nOut; i++){
            for(int j = 0; j < nIn; j++){
                //重み行列の更新
                wIO[i][j] += learningRate * dOutput[i] * input[j] / N;
            }
            //バイアスの更新
            bias[i] += learningRate * dOutput[i] / N;
        }
    }

    /**
     *
     * @param input
     * @param prevInput
     * @param dProjection
     * @param dhOutput
     * @param learningRate
     */
    public void backwardCal2(double input[], double prevInput[],
                             double dProjection[][], double dhOutput[], double learningRate){

        double dOutput[] = new double[nOut];

        //今層の誤差勾配
        for(int i = 0; i < nOut; i++) {
            dOutput[i] = 0;
            //次層の誤差勾配と次層への重み行列との積
            dOutput[i] += dActivation.apply(prevInput[i]) * dhOutput[i];

            for(int n = 0; n < dProjection.length; n++) {
                for (int j = 0; j < nIn; j++) {
                    //活性化関数の微分との積により誤差勾配(修正量)を求める
                    dProjection[n][j] += wIO[i][j] * dOutput[i];
                }
            }

            //今層の誤差勾配を用いて重み行列及びバイアスの更新
            for(int j = 0; j < nIn; j++){
                //重み行列の更新
                wIO[i][j] += learningRate * dOutput[i] * input[j];
            }
            //バイアスの更新
            bias[i] += learningRate * dOutput[i];
        }
    }

    /*
    ここにDropOut
     */
    public void dropout(){

    }
}
