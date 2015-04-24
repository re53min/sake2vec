package org.deeplearning4j.tsne;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.Tsne;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 9/20/14.
 */
public class TsneExample {

    public static void main(String[] args) throws Exception  {
        Tsne tsne = new Tsne.Builder().setMaxIter(10000)
                .learningRate(500).useAdaGrad(false)
                .normalize(false).usePca(false).build();
        Pair<WeightLookupTable,VocabCache> info = WordVectorSerializer.loadTxt(new File(args[0]));
        VocabCache cache = info.getSecond();
        List<String> list = new ArrayList<>();
        for(int i = 0; i < cache.numWords(); i++)
            list.add(cache.wordAtIndex(i));
        InMemoryLookupTable l = (InMemoryLookupTable) info.getFirst();
        INDArray weights = l.getSyn0();
        tsne.plot(weights,2,list);
    }



}
