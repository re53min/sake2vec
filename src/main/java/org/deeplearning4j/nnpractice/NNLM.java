package org.deeplearning4j.nnpractice;

import org.apache.commons.collections.BidiMap;
import org.apache.commons.collections.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.ArrayUtils;
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
 * Created by b1012059 on 2016/01/31.
 */
public class NNLM {
    //private static Logger log = LoggerFactory.getLogger(NNLM.class);
    private int nInput;
    private int nHidden;
    private int nOutput;
    private int vocab;
    private int dim;
    private int n;
    private double learningRate;
    private double decayRate;
    private ProjectionLayer pLayer;
    private HiddenLayer hLayer;
    private LogisticRegression logisticLayer;
    private Random rng;
    private IntToDoubleFunction learningType;

    public NNLM(int N, int vocab, int dim, int n, int nHidden, Random rng, double lr, double dr, String lrUpdateType){
        this.vocab = vocab;
        this.dim = dim;
        this.n = n;
        this.nInput = dim*2;
        this.nHidden = nHidden;
        this.nOutput = vocab;
        this.learningRate = lr;
        this.decayRate = dr;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        this.pLayer = new ProjectionLayer(N, vocab, dim, null, rng);
        this.hLayer = new HiddenLayer(this.nInput, nHidden, null, null, N, rng, "tanh");
        this.logisticLayer = new LogisticRegression(dim, nHidden, this.nOutput, N, rng, "tanh");

        if (lrUpdateType == "UpdateLR" || lrUpdateType == null) {
            this.learningType = (int epoch) -> updateLR(this.learningRate, this.decayRate, epoch);
        } else if(lrUpdateType == "AdaGrad") {
            //this.learningType = (int epoch) -> adaGrad(this.learningRate);
        } else if(lrUpdateType == "RMSProp"){
            //this.learningType = (int epoch) -> rmsProp(this.learningRate);
        } else {
            //log.info("Learning Update Type not supported!");
        }

    }

    public void train(Map<String, Integer> nGram, int epochs, NLP nlp) {
        double[][] lookUpInput = new double[n-1][dim];
        double[] hiddenInput;
        int[] teachInput = new int[vocab];
        double[] outLayerInput;
        double[][] dProjection;
        double[] dhOutput;
        double lr;
        int count = 1;

        /*
        N-gramの袋から着目単語の前n-1単語の分散表現を取り出す
        着目単語のときは教師データの作成
         */
        //log.info("Get LookUpTable and Create TeachData");
        for (Map.Entry<String, Integer> entry : nGram.entrySet()) {
            //log.info("Set " + count + "th N-gram");
            String[] words = entry.getKey().split(" ", 0);
            for (int i = 0; i < n; i++) {
                int vocabNumber = nlp.getWordToId().get(words[i]);
                if (i < n - 1) {
                    lookUpInput[i] = pLayer.lookUpTable(vocabNumber);
                    //log.info("LookUpTable " + vocabNumber + "th word");
                } else {
                    //log.info("TeachData:");
                    teachInput[vocabNumber] = 1;
                }
            }
            /*
            N-gram N回の学習
             */
            //log.info("Training N-gram");
            for (int epoch = 0; epoch < epochs; epoch++) {
                //初期化
                hiddenInput = ArrayUtils.addAll(lookUpInput[0], lookUpInput[1]);
                outLayerInput = new double[nHidden];
                dhOutput = new double[nHidden];
                dProjection = new double[n-1][nInput];
                lr = learningType.applyAsDouble(epoch);
                //log.info(String.valueOf(lr));
                System.out.println(String.valueOf(lr));

                hLayer.forwardCal(hiddenInput, outLayerInput);
                logisticLayer.train2(outLayerInput, lookUpInput, teachInput,
                        dProjection, dhOutput, lr);
                hLayer.backwardCal2(hiddenInput, outLayerInput, dProjection, dhOutput, lr);

                for(int j = 0; j < n - 1; j++) {
                    pLayer.backwardCal(nlp.getWordToId().get(words[j]), dProjection[j]);
                    //LookUpInputの更新
                    lookUpInput[j] = pLayer.lookUpTable(nlp.getWordToId().get(words[j]));
                }
            }
            count++;
        }
        //log.info("Finish N-gram");
    }

    /**
     * テストメソッド
     * @param nlp
     * @param word1
     * @param word2
     * @return
     */
    public void reconstruct(NLP nlp, String word1, String word2){
        double outLayerInput[] = new double[nHidden];
        double output[] = new double[vocab];
        double projection[][] = {pLayer.lookUpTable(nlp.getWordToId().get(word1)),
                pLayer.lookUpTable(nlp.getWordToId().get(word2))};
        BidiMap bidiMap = new DualHashBidiMap(nlp.getWordToId());

        hLayer.forwardCal(ArrayUtils.addAll(pLayer.lookUpTable(nlp.getWordToId().get(word1)),
                pLayer.lookUpTable(nlp.getWordToId().get(word2))), outLayerInput);
        logisticLayer.reconstruct2(outLayerInput, projection, output);

        int index = 0;
        for(int i = 0; i < output.length; i++) index = (output[index] >= output[i]) ? index : i;

        System.out.print((String) bidiMap.getKey(index));
    }

    /**
     * 単語同士のコサイン類似度を求める
     * @param nlp
     * @param word1
     * @param word2
     * @return
     */
    public double cosSim(NLP nlp, String word1, String word2){

        return cosineSimilarity(pLayer.lookUpTable(nlp.getWordToId().get(word1)),
                pLayer.lookUpTable(nlp.getWordToId().get(word2)));
    }

    /**
     * 車輪の再発明
     * DeepLearning4jのWordVectorSerializerクラスにある
     * writeWordVectorsを再実装
     * @param vocab
     * @param dim
     * @param nlp
     */
    public void writeWord(int vocab, int dim, NLP nlp, String fileName){
        writeWordVectors(vocab, dim, nlp, pLayer.getwDI(), fileName);
    }

    private static void testNNLM(){
        /*
        N-gram Size: 111733
        Word Size: 195569
        Vocabulary Size: 10507
         */

        String text = null;
        try {
            text = new String(Files.readAllBytes(Paths.get("target/classes/natsume.txt")), "UTF-8");
        } catch (IOException e) {
            e.printStackTrace();
        }

        /*String text = "ラーメンとは、中華麺とスープ、様々な具（チャーシュー・メンマ・味付け玉子・刻み葱など）を組み合わせた麺料理（ただし具を入れない場合もある）。漢字表記は拉麺、老麺[2]または柳麺。別名は中華そばおよび支那そば・南京そば[3][4]などである。小麦粉を原材料とし、かん水（鹹水）というアルカリ塩水溶液を添加するのが大きな特徴である。そのため同じ小麦粉で作った麺でも、日本のうどんや中国の多くの麺料理と異なる独特の色・味・食感をもつ。"
                +"この小麦粉に水を加えて、細長い麺とする。多くの場合は「製麺機」で製麺し、製麺会社が製造する麺を使用する店も多いが、1990年代以降小型の圧延機などが流通するようになり、ラーメン専門店では自家製麺を行う店が増えている。"
                +"また、麺の太さによって、「細麺」、「中細麺」、「中太麺」、「太麺」などと称する。また、めんの縮れ具合も考慮する。これを組み合わせ、マニアがラーメンの麺を評する際に「中細ストレート麺」などと称することもあるが、あくまでも感覚的な呼称である。博多ラーメンの細い麺からうどんより太い麺まで多種多様である。ラーメンの汁は「スープ」と呼ぶ。丼に入れたタレを出汁（ダシ）で割ってスープを作る（出汁を「スープ」と呼ぶこともあるが、本項では混同を避けるため、区別して記述する）。" +
                "スープはラーメンの味を決定する非常に重要な要素であり、手間暇をかけて工夫したスープを使用する店がほとんどである。そのため、ダシとタレは分けて調理を行う。スープの素となる。出汁は複数の素材から取ることが多く、日本のラーメン原点ともされる醤油ラーメンでは、鶏ガラを基本に、野菜と削り節や煮干しで味を整えたものが主流である。また、「昔風」を標榜しているラーメンも同様のダシを使用することが多い。"
                +"鶏ガラ、豚骨、牛骨、削り節、昆布など様々な材料が、ダシの素材として使用されている。臭み消しにタマネギ、長ネギ、生姜、大蒜などの香味野菜を使う。豚骨をベースにした店も多く、ほかに牛骨や、削り節・煮干し・あごなどの魚介をベースにする店もある。昆布と削り節を組み合わせることで、旨みの相乗効果が生まれることはよく知られている[11]。仏像（ぶつぞう）は、仏教の信仰対象である仏の姿を表現した像のこと。仏（仏陀、如来）の原義は「目覚めた者」で、「真理に目覚めた者」「悟りを開いた者」の意である。初期仏教において「仏」とは仏教の開祖ゴータマ・シッダールタ（釈尊、釈迦如来）を指したが、大乗仏教の発達とともに、弥勒仏、阿弥陀如来などの様々な「仏」の像が造られるようになった。"
                +"「仏像」とは、本来は「仏」の像、すなわち、釈迦如来、阿弥陀如来などの如来像を指すが、一般的には菩薩像、天部像、明王像、祖師像などの仏教関連の像全般を総称して「仏像」ともいう。広義には画像、版画なども含まれるが、一般に「仏像」という時は立体的に表された丸彫りの彫像を指すことが多い。彫像の材質は、金属製、石造、木造、塑造、乾漆造など様々である。元々、釈迦が出世した当時のインド社会では、バラモン教が主流で、バラモン教では祭祀を中心とし神像を造らなかったとされる。当時のインドでは仏教以外にも六師外道などの諸教もあったが、どれも尊像を造って祀るという習慣はなかった。したがって原始仏教もこの社会的背景の影響下にあった。"
                +"また、原始仏教は宗教的側面もあったが、四諦や十二因縁という自然の摂理を観ずる哲学的側面の方がより強かったという理由も挙げられる。さらに釈迦は「自灯明・法灯明」（自らを依り所とし、法を依り所とせよ）という基本的理念から、釈迦本人は、自身が根本的な信仰対象であるとは考えていなかった。したがって初期仏教においては仏像というものは存在しなかった。"
                +"しかし、釈迦が入滅し時代を経ると、仏の教えを伝えるために図画化していくことになる。"
                +"仏陀となった偉大な釈迦の姿は、もはや人の手で表現できないと思われていた。そのため人々は釈迦の象徴としてストゥーパ（卒塔婆、釈迦の遺骨を祀ったもの）、法輪（仏の教えが広まる様子を輪で表現したもの））や、仏足石（釈迦の足跡を刻んだ石）、菩提樹などを礼拝していた。インドの初期仏教美術には仏伝図（釈迦の生涯を表した浮き彫りなど）は多数あるが、釈迦の姿は表されず、足跡、菩提樹、台座などによってその存在が暗示されるのみであった。";
        */

        NLP nlp = new NLP(text);
        int word = nlp.getRet().size();
        int vocab = nlp.getWordToId().size();
        int dim = 30;
        int n = 3;
        Map<String, Integer> map = nlp.createNgram(n);
        int nHidden = 60;
        int epochs = 5;
        double learningRate = 0.1;
        double decayRate = 1E-2;
        Random rng = new Random(123);
        String fileName = "natsume-Model_nnlm";

        /*log.info("Word Size: " + word);
        log.info("Vocabulary Size: " + vocab);
        log.info("Word Vector: " + dim);
        log.info("N-gram Size: " + map.size());
        log.info("Epoch: " + epochs);
        log.info("Learning Rate: " + learningRate);
        log.info("Decay Rate" + decayRate);*/

        //log.info("Creating NNLM Instance");
        NNLM nnlm = new NNLM(word, vocab, dim, n, nHidden, rng, learningRate, decayRate, null);

        //log.info("Starting Train NNLM");
        nnlm.train(map, epochs, nlp);

        //log.info("Saving Word Vectors");
        nnlm.writeWord(vocab, dim, nlp, fileName);

        System.out.println("-------TEST-------");


        Scanner scan = new Scanner(System.in);
        String str = scan.next();

        if (str == null) {
            System.out.println("NULL!!");
        } else switch (str) {
            case "学習":
                nnlm.train(map, epochs, nlp);
            case "類似度":
                Scanner scan1 = new Scanner(System.in);
                String word1 = scan1.next();
                String word2 = scan1.next();
                System.out.println(word1 + "と" + word2 + "のコサイン類似度: " + nnlm.cosSim(nlp, word1, word2));
            case "予測":
                Scanner scan2 = new Scanner(System.in);
                word1 = scan2.next();
                word2 = scan2.next();
                nnlm.reconstruct(nlp, word1, word2);
            case "保存":
                nnlm.writeWord(vocab, dim, nlp, fileName);
            case "終了":
                break;
            default:
        }
        System.out.println("-------FINISH-------");
    }

    public static void main(String[] args){
        //log.info("Let's Start NNLM!!");
        testNNLM();
    }
}
