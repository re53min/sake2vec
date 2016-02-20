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

import static org.deeplearning4j.nnpractice.utils.*;

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
    private double learningRate;
    private double decayRate;
    private RecurrentHLayer rLayer;
    private LogisticRegression logisticLayer;
    private Random rng;
    private IntToDoubleFunction learningType;

    public RNNLM(int N, int vocab, int dim, int nHidden, Random rng, double lr, double dr, String lrUpdateType){
        this.vocab = vocab;
        this.dim = dim;
        this.nInput = vocab;
        this.nHidden = nHidden;
        this.nOutput = vocab;
        this.learningRate = lr;
        this.decayRate = dr;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        this.rLayer = new RecurrentHLayer(vocab, nHidden, null, null, null, null, N, rng, "sigmoid");
        this.logisticLayer = new LogisticRegression(dim, nHidden, this.nOutput, N, rng, "sigmoid");

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

    private void train(Map<String, Integer> nGramm, int epochs, NLP nlp){
        double outLayerInput[];
        double rhLayer[];
        int[] teachInput = new int[vocab];
        int vocabNumber = 0;
        double lr;
        double dOutput[] = new double[vocab];

        log.info("Get LookUpTable and Create TeachData");
        for(int epoch = 0; epoch < epochs; epoch++) {
            for (Map.Entry<String, Integer> entry : nGramm.entrySet()) {
                String[] words = entry.getKey().split(" ", 0);
                for (int i = 0; i < words.length; i++) {
                    if (i < words.length - 1) {
                        vocabNumber = nlp.getWordToId().get(words[i]);
                        log.info("LookUpTable " + vocabNumber + "th word");
                    } else {
                        for (int v = 0; v < vocab; v++) {
                            if (v == nlp.getWordToId().get(words[i])) teachInput[v] = 1;
                            else teachInput[v] = 0;
                        }
                    }
                }

                outLayerInput = new double[nHidden];
                rhLayer = new double[nHidden];

                lr = learningType.applyAsDouble(epoch);
                rLayer.forwardCal(vocabNumber, rhLayer, outLayerInput);
                dOutput = logisticLayer.train(outLayerInput, teachInput, lr);
                rLayer.backwardCal(vocabNumber, null, outLayerInput, dOutput, logisticLayer.wIO, rhLayer, lr);
            }
        }
    }


    private static void testRNNLM(){

        String text = null;
        try {
            text = new String(Files.readAllBytes(Paths.get("target/classes/natsume.txt")), "UTF-8");
        } catch (IOException e) {
            e.printStackTrace();
        }

        NLP nlp = new NLP(text);
        int word = nlp.getRet().size();
        int vocab = nlp.getWordToId().size();
        Map<String, Integer> map = nlp.createNgram(2);
        int dim = 100;
        int nHidden = 100;
        int epochs = 10;
        double learningRate = 0.1;
        double decayRate = 1E-2;
        Random rng = new Random(123);

        log.info("Word size: " + word);
        log.info("Vocabulary size: " + vocab);
        log.info("Word Vector: " + dim);
        log.info("N-gram Size: " + map.size());
        log.info("Epoch: " + epochs);
        log.info("Learning Rate: " + learningRate);
        log.info("Decay Rate" + decayRate);

        log.info("Creating NNLM Instance");
        RNNLM rnnlm = new RNNLM(word, vocab, dim, nHidden, rng, learningRate, decayRate, null);

        log.info("Starting Train NNLM");
        rnnlm.train(map, epochs, nlp);

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
