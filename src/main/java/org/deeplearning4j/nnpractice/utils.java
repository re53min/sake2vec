package org.deeplearning4j.nnpractice;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Created by b1012059 on 2015/09/08.
 */
public class utils {

    //シグモイド関数の傾き
    private final static double beta = 1.0;
    //private static Logger log = LoggerFactory.getLogger(utils.class);

    /**
     * 一様分布
     * 活性化関数がsigmoidの場合はmin,maxにかける4
     * @param nIn
     * @param nOut
     * @param rng
     * @param activation
     * @return
     */
    public static double uniform(int nIn, int nOut, Random rng, String activation){
        double min = -Math.sqrt(6. / (nIn + nOut));
        double max = Math.sqrt(6. / (nIn + nOut));

        if(activation == "sigmoid" || activation == null) {
            min *= 4;
            max *= 4;
        }
        return rng.nextDouble() * (max - min) + min;
    }

    /**
     * 一様分布
     * AutoEncoder用
     * @param nIn
     * @param nOut
     * @param rng
     * @return
     */
    public static double uniform(double nIn, double nOut, Random rng){
        return rng.nextDouble() * (nOut - nIn) + nIn;
    }

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
     * @param tmpOut
     * @return sigmoid
     */
    public static double funSigmoid(double tmpOut){
        double sigmoid = 1.0 / (1.0 + Math.exp(-beta * tmpOut));
        return sigmoid;
    }

    /**
     * シグモイド関数の微分
     * @param tmpOut
     * @return dsigmoid
     */
    public static double dfunSigmoid(double tmpOut){
        double dsigmoid = tmpOut * (1 - tmpOut);
        return dsigmoid;
    }

    /**
     *
     * @param tmpOut
     * @return
     */
    public static double funTanh(double tmpOut){
        double tanh = Math.tanh(tmpOut);
        return tanh;
    }

    /**
     *
     * @param tmpOut
     * @return
     */
    public static double dfunTanh(double tmpOut){
        double dtanh = 1 - tmpOut * tmpOut;
        return dtanh;
    }

    /**
     * Relu(ランプ関数)
     * @param tmpOut
     * @return
     */
    public static double funReLU(double tmpOut){

        if(tmpOut > 0) return tmpOut;
        else return  0.;
    }

    /**
     * Relu(ランプ関数)の微分
     * @param tmpOut
     * @return
     */
    public static double dfunReLU(double tmpOut){

        if(tmpOut > 0) return 1.;
        else return 0.;
    }

    /**
     * ソフトマックス関数
     * @param tmpOut
     * @param nOut
     * @return
     */
    public static void funSoftmax(double tmpOut[], int nOut){
        //double max = 0.0;
        double sum = 0.0;

        /*for(int i = 0; i < nOut; i++){
            if(max < tmpOut[i]) max = tmpOut[i];
        }*/

        for(int i = 0; i < nOut; i++){
            //tmpOut[i] = Math.exp(tmpOut[i] - max);
            tmpOut[i] = Math.exp(tmpOut[i]);
            sum += tmpOut[i];
        }

        for(int i = 0; i < nOut; i++){
            tmpOut[i] /= sum;
        }
    }

    public static double updateLR(double learningRate, double decayRate, int epoch){
        return learningRate / (1 + decayRate * epoch);
    }

    /**
     * AdaGradの実装予定
     * @param learningLate
     * @return
     */
    public static double adaGrad(double learningLate){
        return 0.1;
    }

    /**
     * RMSPropの実装予定
     * @param learningLate
     * @return
     */
    public static double rmsProp(double learningLate){
        return 0.1;
    }

    /**
     * コサイン類似度
     * cosθ = (vectorA*vectorB) / (normA*normB)
     * 分子はベクトルの内積、分母はそれぞれのノルム
     * @param vectorA
     * @param vectorB
     * @return
     */
    public static double cosineSimilarity(double[] vectorA, double[] vectorB){
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for(int i = 0; i < vectorA.length; i++){
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }

        return dotProduct /(Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * 車輪の再発明
     * DeepLearning4jのWordVectorSerializerクラスにある
     * writeWordVectorsを再実装
     * @param vocab
     * @param dim
     * @param nlp
     */
    public static void writeWordVectors(int vocab, int dim, NLP nlp, double w[][], String fileName){
        try {
            BufferedWriter write = new BufferedWriter(new FileWriter(new File(fileName + ".txt"), false));
            boolean flag = true;
            StringBuilder sb = new StringBuilder();
            sb.append(vocab);
            sb.append(" ");
            sb.append(dim);
            sb.append("\n");
            write.write(sb.toString());

            for (String key : nlp.getWordToId().keySet()) {
                sb = new StringBuilder();
                sb.append(key);
                sb.append(" ");

                for(int i = 0; i < dim; i++){
                    sb.append(w[i][nlp.getWordToId().get(key)]);
                    if(i < dim - 1) sb.append(" ");
                }
                sb.append("\n");
                write.write(sb.toString());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
