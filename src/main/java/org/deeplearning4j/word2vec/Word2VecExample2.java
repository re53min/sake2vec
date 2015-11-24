package org.deeplearning4j.word2vec;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;


/**
 * Created by agibsonccc on 10/9/14.
 */
public class Word2VecExample2 {

    private static Logger log = LoggerFactory.getLogger(Word2VecExample2.class);

    public static void main(String[] args) throws Exception {

        Nd4j.getRandom().setSeed(133);

        log.info("Load data....");


        ClassPathResource resource = new ClassPathResource("日本酒コーパスv3.txt");
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

        /*File sakeCorpus = new File("test-words.txt");
        Word2Vec vec = WordVectorSerializer.loadGoogleModel(sakeCorpus, false);*/


        //InMemoryLookupTable table = (InMemoryLookupTable) vec.lookupTable();
        //table.getSyn0().diviRowVector(table.getSyn0().norm2(0));

        log.info("Evaluate model....");

        double[] sim = new double[46];
        sim[0] = vec.similarity("獺祭", "薫酒");
        sim[1] = vec.similarity("久保田", "薫酒");
        sim[2] = vec.similarity("八海山", "薫酒");
        sim[3] = vec.similarity("黒龍", "薫酒");
        sim[4] = vec.similarity("飛露喜", "薫酒");
        sim[5] = vec.similarity("田酒", "薫酒");
        sim[6] = vec.similarity("天狗舞", "薫酒");
        sim[7] = vec.similarity("蓬莱泉", "薫酒");
        sim[8] = vec.similarity("出羽桜", "薫酒");
        sim[9] = vec.similarity("〆張鶴", "薫酒");

        sim[10] = vec.similarity("獺祭", "爽酒");
        sim[11] = vec.similarity("久保田", "爽酒");
        sim[12] = vec.similarity("八海山", "爽酒");
        sim[13] = vec.similarity("黒龍", "爽酒");
        sim[14] = vec.similarity("飛露喜", "爽酒");
        sim[15] = vec.similarity("田酒", "爽酒");
        sim[16] = vec.similarity("天狗舞", "爽酒");
        sim[17] = vec.similarity("蓬莱泉", "爽酒");
        sim[18] = vec.similarity("出羽桜", "爽酒");
        sim[19] = vec.similarity("〆張鶴", "爽酒");

        sim[20] = vec.similarity("獺祭", "醇酒");
        sim[21] = vec.similarity("久保田", "醇酒");
        sim[22] = vec.similarity("八海山", "醇酒");
        sim[23] = vec.similarity("黒龍", "醇酒");
        sim[24] = vec.similarity("飛露喜", "醇酒");
        sim[25] = vec.similarity("田酒", "醇酒");
        sim[26] = vec.similarity("天狗舞", "醇酒");
        sim[27] = vec.similarity("蓬莱泉", "醇酒");
        sim[28] = vec.similarity("出羽桜", "醇酒");
        sim[29] = vec.similarity("〆張鶴", "醇酒");

        sim[30] = vec.similarity("獺祭", "熟酒");
        sim[31] = vec.similarity("久保田", "熟酒");
        sim[32] = vec.similarity("八海山", "熟酒");
        sim[33] = vec.similarity("黒龍", "熟酒");
        sim[34] = vec.similarity("飛露喜", "熟酒");
        sim[35] = vec.similarity("田酒", "熟酒");
        sim[36] = vec.similarity("天狗舞", "熟酒");
        sim[37] = vec.similarity("蓬莱泉", "熟酒");
        sim[38] = vec.similarity("出羽桜", "熟酒");
        sim[39] = vec.similarity("〆張鶴", "熟酒");


        sim[40] = vec.similarity("薫酒", "爽酒");
        sim[41] = vec.similarity("薫酒", "醇酒");
        sim[42] = vec.similarity("薫酒", "熟酒");
        sim[43] = vec.similarity("爽酒", "醇酒");
        sim[44] = vec.similarity("爽酒", "熟酒");
        sim[45] = vec.similarity("醇酒", "熟酒");


        for(int i = 0; i < sim.length; i++){
            if(i % 10 == 0){
                System.out.println("-------------------------");
                System.out.println(sim[i]);
            } else {
                System.out.println(sim[i]);
            }
        }
        System.out.println("-------------------------");

        /*Collection<String> similar = vec.wordsNearest("山廃" , 20);
        log.info("Similar words to: " + similar);

        String[] sake = {"薫酒","爽酒","醇酒","熟酒"};
        String[] tmpWord = similar.toArray(new String[0]);
        log.info(tmpWord[0]);
        double[] tmpsim = new double[4];
        for(int i = 0; i < sake.length; i++){
            tmpsim[i] = vec.similarity(sake[i], tmpWord[0]);
            log.info("Similarity between" + sake[i] + "and " + tmpWord[0] + ": "+ tmpsim[i]);
        }
        List simList = Arrays.asList(tmpsim);
        Collections.sort(simList);
        Collections.reverse(simList);
        log.info(Arrays.toString(tmpsim));*/


        /*Collection<String> operation = vec.wordsNearest(Arrays.asList("獺祭", "辛い"), Arrays.asList("甘い"), 10);
        log.info("Word operation: " + operation);*/


        //log.info("Save vectors....");
        //WordVectorSerializer.writeWordVectors(vec, "test-words.txt");

        /*log.info("Plot TSNE.....");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .usePca(false)
                .build();
        vec.lookupTable().plotVocab(tsne);*/

        log.info("****************Example finished********************");


    }

}