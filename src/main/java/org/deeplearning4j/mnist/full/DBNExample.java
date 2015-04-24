package org.deeplearning4j.mnist.full;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.instrumentation.server.InstrumentationApplication;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;

/**
 * Created by agibsonccc on 9/11/14.
 */
public class DBNExample {

    private static Logger log = LoggerFactory.getLogger(DBNExample.class);


    public static void main(String[] args) throws Exception {
        Nd4j.dtype = DataBuffer.FLOAT;
//        Nd4j.shouldInstrument = true;
//        Thread t = new Thread(new Runnable() {
//            @Override
//            public void run() {
//                InstrumentationApplication app = new InstrumentationApplication();
//                app.start();
//            }
//        });
//        t.start();


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.5).layerFactory(LayerFactories.getFactory(RBM.class))
                .momentumAfter(Collections.singletonMap(3, 0.9)).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .iterations(5).weightInit(WeightInit.DISTRIBUTION).dist(Nd4j.getDistributions().createUniform(0, 1)).iterationListener(new ScoreIterationListener(1))
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(784).nOut(10).list(4)
                .hiddenLayerSizes(new int[]{500, 250, 200})
                .override(new ClassifierOverride(3))
                .build();


        MultiLayerNetwork d = new MultiLayerNetwork(conf);
        DataSetIterator iter = new MnistDataSetIterator(100,60000);
        while(iter.hasNext()) {
            DataSet next = iter.next();
            d.fit(next);

        }


        iter.reset();

        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {

            DataSet d2 = iter.next();
            INDArray predict2 = d.output(d2.getFeatureMatrix());

            eval.eval(d2.getLabels(), predict2);

        }

        log.info(eval.stats());


    }

}
