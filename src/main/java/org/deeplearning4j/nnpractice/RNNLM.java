package org.deeplearning4j.nnpractice;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.function.IntToDoubleFunction;

import static org.deeplearning4j.nnpractice.utils.adaGrad;
import static org.deeplearning4j.nnpractice.utils.rmsProp;
import static org.deeplearning4j.nnpractice.utils.updateLR;

/**
 * Created by b1012059 on 2016/02/15.
 */
public class RNNLM {
    private static Logger log = LoggerFactory.getLogger(RNNLM.class);
    private int nInput;
    private int nHidden;
    private int nOutput;
    private int vocab;
    private int dim;
    private int n;
    private double learningRate;
    private double decayRate;
    private RecurrentHLayer rLayer;
    private HiddenLayer hLayer;
    private LogisticRegression logisticLayer;
    private Random rng;
    private IntToDoubleFunction learningType;

    public RNNLM(int vocab, int dim, int n, int nHidden, Random rng, double lr, double dr, String lrUpdateType){
        this.vocab = vocab;
        this.dim = dim;
        this.n = n;
        this.nInput = vocab;
        this.nHidden = nHidden;
        this.nOutput = vocab;
        this.learningRate = lr;
        this.decayRate = dr;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //this.rLayer = new RecurrentHLayer();
        this.hLayer = new HiddenLayer(this.nInput, nHidden, null, null, vocab, rng, "tanh");
        this.logisticLayer = new LogisticRegression(dim, nHidden, this.nOutput, vocab, rng, "tanh");

        if (lrUpdateType == "UpdateLR" || lrUpdateType == null) {
            this.learningType = (int epoch) -> updateLR(this.learningRate, this.decayRate, epoch);
        } else if(lrUpdateType == "AdaGrad") {
            this.learningType = (int epoch) -> adaGrad(this.learningRate);
        } else if(lrUpdateType == "RMSProp"){
            this.learningType = (int epoch) -> rmsProp(this.learningRate);
        } else {
            log.info("Learning Update Type not supported!");
        }
    }

    private void train(){

    }


    private static void testRNNLM(){

        String text = null;
        try {
            text = new String(Files.readAllBytes(Paths.get("target/classes/natsume.txt")), "UTF-8");
        } catch (IOException e) {
            e.printStackTrace();
        }

        NLP nlp = new NLP(text);
        int vocab = nlp.getWordToId().size();
        int dim = 30;
        int n = 3;
        Map<String, Integer> map = nlp.createNgram(n);
        int nHidden = 60;
        int epochs = 10;
        double learningRate = 0.1;
        double decayRate = 1E-2;
        Random rng = new Random(123);

        log.info("Word size: " + nlp.getRet().size());
        log.info("Vocabulary size: " + vocab);
        log.info("Word Vector: " + dim);
        log.info("N-gram Count: " + map.size());
        log.info("Epoch: " + epochs);
        log.info("Learning Rate: " + learningRate);
        log.info("Decay Rate" + decayRate);

        log.info("Creating NNLM Instance");
        RNNLM rnnlm = new RNNLM(vocab, dim, n, nHidden, rng, learningRate, decayRate, null);

        log.info("Starting Train NNLM");
        //rnnlm.train(map, epochs, nlp);

        System.out.println("-------TEST-------");


        Scanner scan = new Scanner(System.in);
        String str = scan.next();

        /*
        if (str == null) {
            System.out.println("NULL!!");
        } else switch (str) {
            case "学習":
                rnnlm.train(map, epochs, nlp);
            case "類似度":
                Scanner scan1 = new Scanner(System.in);
                String word1 = scan1.next();
                String word2 = scan1.next();
                System.out.println(word1 + "と" + word2 + "のコサイン類似度: " + rnnlm.cosSim(nlp, word1, word2));
            case "予測":
                Scanner scan2 = new Scanner(System.in);
                word1 = scan2.next();
                word2 = scan2.next();
                rnnlm.reconstruct(nlp, word1, word2);
            case "終了":
                break;
            default:
        }
        System.out.println("-------FINISH-------");
        */

    }

    public static void main(String args[]){
        log.info("Let's Start NNLM!!");
        testRNNLM();
    }
}
