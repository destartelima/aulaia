import java.io.FileReader;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;


public class AlgoritmoKnn {
	public static void main (String [] args) throws Exception{
		
		FileReader leitor= new FileReader("iris.arff");
		Instances iris = new Instances(leitor);
		iris.setClassIndex(4);
		Instance exemplo = new Instance(5);
		
		exemplo.setDataset(iris);
		//copiar dados do primeiro exemplo após o @data e remover do arquivo arff
		
		exemplo.setValue(0,5.1);
		exemplo.setValue(1,3.5);
		exemplo.setValue(2,1.4);
		exemplo.setValue(3,0.2);
		
		IBk knn = new IBk(3);
		knn.buildClassifier(iris);//aprendizado
		double classe = knn.classifyInstance(exemplo);
		System.out.println("Exemplo: "+exemplo);
		System.out.println("Resposta:"+ classe);
		
	}

}
