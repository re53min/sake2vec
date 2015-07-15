package org.deeplearning4j.word2vec;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.core.io.ClassPathResource;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


/**
 * Created by agibsonccc on 10/9/14.
 */
public class Word2VecExample {


    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        //DocumentIterator iter = new FileDocumentIterator(resource.getFile());
        TokenizerFactory t = new DefaultTokenizerFactory();
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
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

        int layerSize = 300;

        Word2Vec vec = new Word2Vec.Builder().sampling(1e-5)
                .minWordFrequency(5).batchSize(1000).useAdaGrad(false).layerSize(layerSize)
                .iterations(3).learningRate(0.025).minLearningRate(1e-2).negativeSample(10)
                .iterate(iter).tokenizerFactory(t).build();
        vec.fit();

        //similarity(string　A, string　B):AとBの近似値
        double sim = vec.similarity("people", "money");
        System.out.println("Similarity between people and money " + sim);

        //wordsNearest(string　A, int　N):Aに近い単語をN個抽出
        Collection<String> similar = vec.wordsNearest("money", 10);
        System.out.println("wordNearest:" + similar);
        List<String> tmpData = (List<String>) similar;
        for(int i = 0; i < similar.size(); i++){
            double sim2 = vec.similarity("money", tmpData.get(i));
            System.out.println("money and " + tmpData.get(i)+ " is " + sim2);
        }

        /*//test wordsNearestSum
        Collection<String> testWordNearestSum = vec.wordsNearestSum("money", 20);
        System.out.println("WordNearestSum:" + testWordNearestSum);*/

        //test similarWordsInVocabTo
        Collection<String> testSimilarWordInVocabTo = vec.similarWordsInVocabTo("money", 0.7);
        System.out.println("SimilarWordInVocabTo:" + testSimilarWordInVocabTo);

        //test getWordVector
        double[] testGetWordVector = vec.getWordVector("money");
        System.out.println("GetWordVector:" + testGetWordVector[0]);

        //test getWordVectorMatrixNormalized
        //INDArray testGetWordVectorMatrixNormalized = vec.getWordVectorMatrixNormalized("money");
        //System.out.println("GetWordVectorMatrixNormalized:" + testGetWordVectorMatrixNormalized);

        //test getWordVectorMatrix
        //INDArray testGetWordVectorMatrix = vec.getWordVectorMatrix("money");
        //System.out.println("GetWordVectorMatrix:" + testGetWordVectorMatrix);

        //test wordNearest(String positive, String negative, int top)
        List<String> list1 = (List<String>) similar;
        //List<String> list1 = new ArrayList();
        List<String> list2 = new ArrayList();
        //list1.add("money");
        list2.add("work");
        Collection<String> testWordNearest = vec.wordsNearest(list1, list2, 10);
        System.out.println("testWordNearest(positive, negative, top):" + testWordNearest);


        Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(200).useAdaGrad(false)
                .normalize(false).usePca(false).build();


        //vec.lookupTable().plotVocab(tsne);



    }

}
