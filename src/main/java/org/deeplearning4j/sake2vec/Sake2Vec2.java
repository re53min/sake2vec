package org.deeplearning4j.sake2vec;

/**
 * sake2vec2本体
 * Created by b1012059 on 2015/04/25.
 * @auther Wataru Matsudate
 */

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

class Sake2Vec2 {
    private String fileName, modelName;
    private String word1;
    private String word2;
    private Word2Vec vec;
    private boolean flag;
    private File sakeCorpus, googleNV;
    private static Logger log = LoggerFactory.getLogger(Sake2Vec2.class);


    /**
     *
     * @param fileName
     * @param word1
     * @param word2
     */
    public Sake2Vec2(String fileName, String word1, String word2) {

        this.fileName = fileName;
        this.word1 = word1;
        this.word2 = word2;

    }

    /**
     *
     * @param word1
     * @param word2
     */
    public Sake2Vec2(String word1, String word2){

        this.word1 = word1;
        this.word2 = word2;

    }

    /**
     *
     * @param fileName
     */
    public Sake2Vec2(String fileName, boolean flag){

        this.flag = flag;
        if(flag) this.modelName = fileName;
        else this.fileName = fileName;

    }

    /**
     *
     */
    public Sake2Vec2(){

    }

    /**
     *
     * @throws Exception
     */
    public void runSake2vec2() throws Exception {

        //googleNV = new File("GoogleNews-vectors-negative300.bin.gz");

        if(flag){
            sakeCorpus = new File(modelName);
            log.info("Exist word vectors model. Reload model...");
            vec = WordVectorSerializer.loadGoogleModel(sakeCorpus, false);

            /*WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(sakeCorpus);
            WeightLookupTable weightLookupTable = wordVectors.lookupTable();
            Iterator<INDArray> vectors = weightLookupTable.vectors();
            INDArray wordVector = vectors.getWordVectorMatrix("myword");
            double[] wordVector = vectors.getWordVector("myword");*/

            log.info("****************Reload model finished********************");

        } else {
            log.info("Not Exist word vectors model. Load data...");
            ClassPathResource resource = new ClassPathResource(fileName);
            //ClassPathResource resource = new ClassPathResource("日本酒コーパス.txt");
            SentenceIterator iter = new LineSentenceIterator(resource.getFile());
            iter.setPreProcessor(new SentencePreProcessor() {
                @Override
                public String preProcess(String sentence) {
                    return sentence.toLowerCase();
                }
            });

            log.info("Tokenize data....");
            final EndingPreProcessor preProcessor = new EndingPreProcessor();
            TokenizerFactory tokenizer = new DefaultTokenizerFactory();
            tokenizer.setTokenPreProcessor(new TokenPreProcess() {
                @Override
                public String preProcess(String token) {
                    token = token.toLowerCase();
                    String base = preProcessor.preProcess(token);
                    //base = base.replaceAll("\\d", "d");
                    if (base.endsWith("ly") || base.endsWith("ing"))
                        System.out.println();
                    return base;
                }
            });

            // Customizing params
            int batchSize = 1000;
            int iterations = 30;
            int layerSize = 30;

            log.info("Build model....");
            Word2Vec vec = new Word2Vec.Builder()
                    .batchSize(batchSize)           // words per minibatch
                    .sampling(1e-5)                 // sub sampling. drops words out
                    .minWordFrequency(2)            // min word frequency
                    .useAdaGrad(false)              // use AdaGrad. in case, not use
                    .layerSize(layerSize)           // words feature vector size
                    .iterations(iterations)         // iterations to train
                    .learningRate(0.025)            // learning rate
                    .minLearningRate(1e-2)          // learning rate decays wrt #words. floor learning
                    .negativeSample(15)              // negative sampling size n words
                    .iterate(iter)                  // learn words batch
                    .windowSize(15)                 // window size
                    .tokenizerFactory(tokenizer)    // create tokenizer
                    .build();                       // build
            vec.fit();

            //InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
            //table.getSyn0().diviRowVector(table.getSyn0().norm2(0));

            log.info("save model");
            WordVectorSerializer.writeWordVectors(vec, fileName);
            modelName = "words.txt";

            //ここにバグがあるよ
            this.flag = true;

            log.info("****************Build model finished********************");

        }
    }


    /**
     *
     * @param word1
     * @param word2
     * @return
     * @throws Exception
     */
    public double sakeSimilar(String word1,  String word2) throws Exception {
        double result = 0.0;
        double sim;
        if(vec != null){
            try {
                log.info("***********************vec exists***********************");
                //similarity(string　A, string　B):AとBの近似値
                sim = vec.similarity(word1, word2);
                result = sim;
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            try {
                log.info("***********************vec null***********************");
                runSake2vec2();
                sim = vec.similarity(word1, word2);
                result = sim;
            } catch (Exception e){
                e.printStackTrace();
            }
        }
        return result;
    }

    /**
     *
     * @param word
     * @param number
     * @return
     * @throws Exception
     */
    public Collection<String> sakeWordsNearest(String word, int number) throws Exception{
        return this.sakeWordsNearest(Arrays.asList(new String[]{word}), new ArrayList(), number);
    }


    /**
     *
     * @param posi
     * @param nega
     * @param number
     * @return
     */
    public Collection<String> sakeWordsNearest(List<String> posi, List<String> nega, int number) throws Exception {
        Collection<String> result = null;
        Collection<String> similar;

        if(vec != null){
            try {
                //wordsNearest(string　A, int　N):Aに近い単語をN個抽出
                similar = vec.wordsNearest(posi, nega, number);
                //System.out.println(similar);
                result = similar;
            } catch (Exception e){
                e.printStackTrace();
            }
        } else {
            try {
                runSake2vec2();
                similar = vec.wordsNearest(posi, nega, number);
                result = similar;
            } catch (Exception e){
                e.printStackTrace();
            }
        }
        return result;
    }

}
