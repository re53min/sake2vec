package org.deeplearning4j.nnpractice;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Random;
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

    public RNNLM(int N, int vocab, int dim, Random rng, double lr, double dr, String lrUpdateType){
        this.vocab = vocab;
        this.dim = dim;
        this.nInput = vocab;
        this.nHidden = dim;
        this.nOutput = vocab;
        this.learningRate = lr;
        this.decayRate = dr;

        //randomの種
        if(rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        this.rLayer = new RecurrentHLayer(vocab, nHidden, null, null, null, null, N, rng, "sigmoid");
        this.logisticLayer = new LogisticRegression(nHidden, this.nOutput, N, rng, "sigmoid");

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
        double rhInput[] = new double[nHidden];
        int[] teachInput = new int[vocab];
        int vocabNumber = 0;
        double lr = learningRate;
        double dOutput[];

        log.info("Get LookUpTable and Create TeachData");
        for(int epoch = 0; epoch < epochs; epoch++) {
            log.info(String.valueOf(lr));
            for (Map.Entry<String, Integer> entry : nGramm.entrySet()) {
                String[] words = entry.getKey().split(" ", 0);
                for (int i = 0; i < words.length; i++) {
                    if (i < words.length - 1) {
                        vocabNumber = nlp.getWordToId().get(words[i]);
                        //log.info("LookUpTable " + vocabNumber + "th word");
                    } else {
                        for (int v = 0; v < vocab; v++) {
                            if (v == nlp.getWordToId().get(words[i])) teachInput[v] = 1;
                            else teachInput[v] = 0;
                        }
                    }
                }

                outLayerInput = new double[nHidden];
                lr = learningType.applyAsDouble(epoch);

                rLayer.forwardCal(vocabNumber, rhInput, outLayerInput);
                dOutput = logisticLayer.train(outLayerInput, teachInput, lr);
                rLayer.backwardCal(vocabNumber, null, outLayerInput, dOutput, logisticLayer.wIO, rhInput, lr);

                rhInput = outLayerInput;
            }
        }
    }

    /**
     * 単語同士のコサイン類似度を求める
     * @param nlp
     * @param word1
     * @param word2
     * @return
     */
    public double cosSim(NLP nlp, String word1, String word2){

        return cosineSimilarity(rLayer.lookUpTable(nlp.getWordToId().get(word1)),
                rLayer.lookUpTable(nlp.getWordToId().get(word2)));
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
        writeWordVectors(vocab, dim, nlp, rLayer.getwIH(), fileName);
    }


    private static void testRNNLM(){

        /*String text = null;
        try {
            text = new String(Files.readAllBytes(Paths.get("target/classes/natsume.txt")), "UTF-8");
        } catch (IOException e) {
            e.printStackTrace();
        }*/

        String text = "ラーメンとは、中華麺とスープ、様々な具（チャーシュー・メンマ・味付け玉子・刻み葱など）を組み合わせた麺料理（ただし具を入れない場合もある）。漢字表記は拉麺、老麺[2]または柳麺。別名は中華そばおよび支那そば・南京そば[3][4]などである。小麦粉を原材料とし、かん水（鹹水）というアルカリ塩水溶液を添加するのが大きな特徴である。そのため同じ小麦粉で作った麺でも、日本のうどんや中国の多くの麺料理と異なる独特の色・味・食感をもつ。"
                +"この小麦粉に水を加えて、細長い麺とする。多くの場合は「製麺機」で製麺し、製麺会社が製造する麺を使用する店も多いが、1990年代以降小型の圧延機などが流通するようになり、ラーメン専門店では自家製麺を行う店が増えている。"
                +"また、麺の太さによって、「細麺」、「中細麺」、「中太麺」、「太麺」などと称する。また、めんの縮れ具合も考慮する。これを組み合わせ、マニアがラーメンの麺を評する際に「中細ストレート麺」などと称することもあるが、あくまでも感覚的な呼称である。博多ラーメンの細い麺からうどんより太い麺まで多種多様である。ラーメンの汁は「スープ」と呼ぶ。丼に入れたタレを出汁（ダシ）で割ってスープを作る（出汁を「スープ」と呼ぶこともあるが、本項では混同を避けるため、区別して記述する）。" +
                "スープはラーメンの味を決定する非常に重要な要素であり、手間暇をかけて工夫したスープを使用する店がほとんどである。そのため、ダシとタレは分けて調理を行う。スープの素となる。出汁は複数の素材から取ることが多く、日本のラーメン原点ともされる醤油ラーメンでは、鶏ガラを基本に、野菜と削り節や煮干しで味を整えたものが主流である。また、「昔風」を標榜しているラーメンも同様のダシを使用することが多い。"
                +"鶏ガラ、豚骨、牛骨、削り節、昆布など様々な材料が、ダシの素材として使用されている。臭み消しにタマネギ、長ネギ、生姜、大蒜などの香味野菜を使う。豚骨をベースにした店も多く、ほかに牛骨や、削り節・煮干し・あごなどの魚介をベースにする店もある。昆布と削り節を組み合わせることで、旨みの相乗効果が生まれることはよく知られている[11]。仏像（ぶつぞう）は、仏教の信仰対象である仏の姿を表現した像のこと。仏（仏陀、如来）の原義は「目覚めた者」で、「真理に目覚めた者」「悟りを開いた者」の意である。初期仏教において「仏」とは仏教の開祖ゴータマ・シッダールタ（釈尊、釈迦如来）を指したが、大乗仏教の発達とともに、弥勒仏、阿弥陀如来などの様々な「仏」の像が造られるようになった。"
                +"「仏像」とは、本来は「仏」の像、すなわち、釈迦如来、阿弥陀如来などの如来像を指すが、一般的には菩薩像、天部像、明王像、祖師像などの仏教関連の像全般を総称して「仏像」ともいう。広義には画像、版画なども含まれるが、一般に「仏像」という時は立体的に表された丸彫りの彫像を指すことが多い。彫像の材質は、金属製、石造、木造、塑造、乾漆造など様々である。元々、釈迦が出世した当時のインド社会では、バラモン教が主流で、バラモン教では祭祀を中心とし神像を造らなかったとされる。当時のインドでは仏教以外にも六師外道などの諸教もあったが、どれも尊像を造って祀るという習慣はなかった。したがって原始仏教もこの社会的背景の影響下にあった。"
                +"また、原始仏教は宗教的側面もあったが、四諦や十二因縁という自然の摂理を観ずる哲学的側面の方がより強かったという理由も挙げられる。さらに釈迦は「自灯明・法灯明」（自らを依り所とし、法を依り所とせよ）という基本的理念から、釈迦本人は、自身が根本的な信仰対象であるとは考えていなかった。したがって初期仏教においては仏像というものは存在しなかった。"
                +"しかし、釈迦が入滅し時代を経ると、仏の教えを伝えるために図画化していくことになる。"
                +"仏陀となった偉大な釈迦の姿は、もはや人の手で表現できないと思われていた。そのため人々は釈迦の象徴としてストゥーパ（卒塔婆、釈迦の遺骨を祀ったもの）、法輪（仏の教えが広まる様子を輪で表現したもの））や、仏足石（釈迦の足跡を刻んだ石）、菩提樹などを礼拝していた。インドの初期仏教美術には仏伝図（釈迦の生涯を表した浮き彫りなど）は多数あるが、釈迦の姿は表されず、足跡、菩提樹、台座などによってその存在が暗示されるのみであった。";


        NLP nlp = new NLP(text);
        int word = nlp.getRet().size();
        int vocab = nlp.getWordToId().size();
        Map<String, Integer> map = nlp.createNgram(2);
        int dim = 100;
        int epochs = 5;
        double learningRate = 0.1;
        double decayRate = 0.95;
        Random rng = new Random(123);
        String fileName = "natsume-Model_rnnlm";

        log.info("Word size: " + word);
        log.info("Vocabulary size: " + vocab);
        log.info("Word Vector: " + dim);
        log.info("N-gram Size: " + map.size());
        log.info("Epoch: " + epochs);
        log.info("Learning Rate: " + learningRate);
        log.info("Decay Rate" + decayRate);

        log.info("Creating NNLM Instance");
        RNNLM rnnlm = new RNNLM(word, vocab, dim, rng, learningRate, decayRate, null);

        log.info("Starting Train NNLM");
        rnnlm.train(map, epochs, nlp);

        log.info("Saving Word Vectors");
        rnnlm.writeWord(vocab, dim, nlp, fileName);

        /*System.out.println("-------TEST-------");

        Scanner scan = new Scanner(System.in);
        String str = scan.next();

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
        */
        System.out.println("-------FINISH-------");

    }

    public static void main(String args[]){
        log.info("Let's Start RNNLM!!");
        testRNNLM();
    }
}
