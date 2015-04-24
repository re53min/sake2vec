package org.deeplearning4j.reconstruct;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by agibsonccc on 9/11/14.
 */
public class DBNExample {

    private static Logger log = LoggerFactory.getLogger(DBNExample.class);


    public static void main(String[] args) throws Exception {
        RandomGenerator gen = new MersenneTwister(123);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.VI)
                .iterations(5).layerFactory(LayerFactories.getFactory(RBM.class))
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .learningRate(1e-1f).nIn(784).nOut(10).list(4)
                .hiddenLayerSizes(new int[]{600, 500, 400})
                .override(new ClassifierOverride(3))
                .build();




        MultiLayerNetwork network = new MultiLayerNetwork(conf);


        DataSetIterator iter = new MultipleEpochsIterator(10,new MnistDataSetIterator(1000,1000));
        network.fit(iter);


        iter.reset();

        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {

            DataSet d2 = iter.next();
            INDArray predict2 = network.output(d2.getFeatureMatrix());

            eval.eval(d2.getLabels(), predict2);

        }

        log.info(eval.stats());


    }

}
