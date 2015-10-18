package org.deeplearning4j.word2vec;

/**
 * sake2vec2本体
 * Created by b1012059 on 2015/04/25.
 * @auther Wataru Matsudate
 */
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
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

class Sake2Vec2 extends Sake2Vec{
    private String fileName, modelName;
    private String word1;
    private String word2;
    private Word2Vec vec;
    private boolean flag;
    private File sakeCorpus, googleNV;
    private static Logger log = LoggerFactory.getLogger(Sake2Vec.class);


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
    public void runSake2vec() throws Exception {

        //googleNV = new File("GoogleNews-vectors-negative300.bin.gz");

        if(flag){
            sakeCorpus = new File(modelName);
            log.info("Exist word vectors model. Reload model...");
            vec = WordVectorSerializer.loadGoogleModel(sakeCorpus, false);

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
            TokenizerFactory t = new DefaultTokenizerFactory();
            t.setTokenPreProcessor(new TokenPreProcess() {
                @Override
                public String preProcess(String token) {
                    token = token.toLowerCase();
                    String base = preProcessor.preProcess(token);
                    base = base.replaceAll("\\d", "d");
                    if (base.endsWith("ly") || base.endsWith("ing"))
                        System.out.println();
                    return base;
                }
            });

            int batchSize = 1000;
            int iterations = 30;
            int layerSize = 50;

            log.info("Build model...");
            vec = new Word2Vec.Builder()
                    .batchSize(batchSize) //# words per minibatch.
                    .sampling(1e-5) // negative sampling. drops words out
                    .minWordFrequency(5) //
                    .useAdaGrad(false) //
                    .layerSize(layerSize) // word feature vector size
                    .iterations(iterations) // # iterations to train
                    .learningRate(0.025) //
                    .minLearningRate(1e-2) // learning rate decays wrt # words. floor learning
                    .negativeSample(10) // sample size 10 words
                    .iterate(iter) //
                    .tokenizerFactory(t)
                    .build();
            vec.fit();

            InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
            table.getSyn0().diviRowVector(table.getSyn0().norm2(0));

            log.info("save model");
            WordVectorSerializer.writeWordVectors(vec, "words.txt");
            modelName = "words.txt";
            flag = true;

            log.info("****************Build model finished********************");

        }
    }


}