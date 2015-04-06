

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeSet;

public class NaiveBayesTrain {

	static Integer spamFileCount=0,hamFileCount=0,N=0,classes=0;
	static double prior[];
	static ArrayList<String> vList=new ArrayList<String>();
	static String[] hamTest,spamTest;
	static String hamTestFile,spamTestFile,hamTrainFile,spamTrainFile;
	public static void main(String[] args) throws Exception {
		hamTrainFile=args[0];
		spamTrainFile=args[1];
		hamTestFile=args[2];
		spamTestFile=args[3];
		Double[][] condprob = bayesTraining();
		System.out.println("Ham Accuracy is:"+hamTest(condprob));
		System.out.println("Spam Accuracy is:"+spamTest(condprob));
	}

	private static double hamTest(Double[][] condprob)
			throws FileNotFoundException, IOException {
		File hamTestfolder = new File(hamTestFile);
		File[] hamTestlistOfFiles = hamTestfolder.listFiles();
		hamTest=new String[hamTestlistOfFiles.length];
		StringBuilder builder=new StringBuilder(); 
		int posCount=0;
		for (int i = 0; i < hamTestlistOfFiles.length; i++) {
			builder=new StringBuilder();
			FileInputStream en=new FileInputStream(new File(hamTestlistOfFiles[i].toString()));
			BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(hamTestlistOfFiles[i].toString()))));
			String str=x.readLine();
			while (str!=null) {
				builder.append(str);
				str=x.readLine();
			}
			String strTemp=builder.toString().replaceAll("[^a-zA-Z]"," ");
			strTemp=builder.toString().replaceAll("\\s+", " ");
			double score[]=new double[classes];
			for (int j = 0; j < classes; j++) {
				score[j]=(double)(-1*Math.log10(prior[j])/Math.log(2));
				String[] tempArray=strTemp.split(" ");
				for (int k = 0; k < tempArray.length; k++) {
					String t=tempArray[k];
					int index=vList.indexOf(t);
					if(index!=-1){
						score[j]+=(double)(-1*Math.log10(condprob[index][j])/Math.log(2));
					}
				}
			}
			if(score[0]<score[1]){
				hamTest[i]="HAM";
				posCount++;
			}
			else{
				hamTest[i]="SPAM";
			}
		}
		double x=(double)posCount/(double)hamTestlistOfFiles.length;
		return x;
	}

	private static double spamTest(Double[][] condprob)
			throws FileNotFoundException, IOException {
		File hamTestfolder = new File(spamTestFile);
		File[] hamTestlistOfFiles = hamTestfolder.listFiles();
		spamTest=new String[hamTestlistOfFiles.length];
		int posCount=0;
		StringBuilder builder=new StringBuilder(); 
		for (int i = 0; i < hamTestlistOfFiles.length; i++) {
			builder=new StringBuilder();
			FileInputStream en=new FileInputStream(new File(hamTestlistOfFiles[i].toString()));
			BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(hamTestlistOfFiles[i].toString()))));
			String str=x.readLine();
			while (str!=null) {
				builder.append(str);
				str=x.readLine();
			}
			String strTemp=builder.toString().replaceAll("[^a-zA-Z]"," ");
			strTemp=builder.toString().replaceAll("\\s+", " ");
			double score[]=new double[classes];
			for (int j = 0; j < classes; j++) {
				score[j]=(double)(-1*Math.log(prior[j])/Math.log(2));
				String[] tempArray=strTemp.split(" ");
				for (int k = 0; k < tempArray.length; k++) {

					String t=tempArray[k];
					int index=vList.indexOf(t);
					if(index!=-1){
						score[j]+=(double)(-1*Math.log(condprob[index][j])/Math.log(2));
					}	
				}
			}
			if(score[0]<score[1]){
				spamTest[i]="HAM";
			}
			else{
				spamTest[i]="SPAM";
				posCount++;
			}
		}
		double x=(double)posCount/(double)hamTestlistOfFiles.length;
		return x;
	}
	private static Double[][] bayesTraining() throws Exception{
		ArrayList<String[]> vocab=extractVocab();
		classes=vocab.size();
		String[] ham=vocab.get(0);
		String[] spam=vocab.get(1);
		TreeSet<String> V=new TreeSet<String>();
		for (int i = 0; i < ham.length; i++) {
			ham[i]=ham[i].replaceAll("[^a-zA-Z]"," ");
			ham[i]=ham[i].replaceAll("\\s+", " ");
			String temp[]=ham[i].split(" ");
			for (int j = 0; j < temp.length; j++) {
				V.add(temp[j]);
			}
		}
		for (int i = 0; i < spam.length; i++) {
			spam[i]=spam[i].replaceAll("[^a-zA-Z]"," ");
			spam[i]=spam[i].replaceAll("\\s+", " ");

			String temp[]=spam[i].split(" ");
			for (int j = 0; j < temp.length; j++) {
				V.add(temp[j]);
			}
		}
		System.out.println("VSize is:"+V.size());
		spamFileCount=spam.length;
		hamFileCount=ham.length;
		N=spamFileCount+hamFileCount;
		String[] str=new String[vocab.size()];
		prior=new double[vocab.size()];
		StringBuilder textc=new StringBuilder();

		Double[][] condprob=new Double[V.size()][vocab.size()]; 
		for (int i = 0; i < vocab.size(); i++) {//for each class
			Integer Nc=vocab.get(i).length;
			textc=new StringBuilder();
			prior[i]=(double)Nc/(double)N;//calculate prior value for this class
			if(i==0){//for ham,calculate textc
				File folder = new File(hamTrainFile);
				TreeSet<String> hamSet=new TreeSet<String>();
				File[] hamlistOfFiles = folder.listFiles();
				StringBuilder builder=new StringBuilder(); 
				for (int j = 0; j < hamlistOfFiles.length; j++) {
					FileInputStream en=new FileInputStream(new File(hamlistOfFiles[j].toString()));
					BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(hamlistOfFiles[j].toString()))));
					String str1=x.readLine();
					while (str1!=null) {
						builder.append(str1);
						str1=x.readLine();
					}
				}
				str[i]=builder.toString().replaceAll("[^a-zA-Z]"," ");
				str[i]=str[i].replaceAll("\\s+", " ");
				String[] temp=str[i].split(" ");
				for (int j = 0; j < temp.length; j++) {
					boolean b=hamSet.add(temp[j]);
					if(b==true||b==false){
						textc.append(temp[j]+" ");
					}
				}
				System.out.println("Hamset count:"+hamSet.size());
			}
			else if(i==1){
				File folder = new File(spamTrainFile);
				File[] spamlistOfFiles = folder.listFiles();
				StringBuilder builder=new StringBuilder(); 
				for (int j = 0; j < spamlistOfFiles.length; j++) {
					FileInputStream en=new FileInputStream(new File(spamlistOfFiles[j].toString()));
					BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(spamlistOfFiles[j].toString()))));
					String str1=x.readLine();
					while (str1!=null) {
						builder.append(str1);
						str1=x.readLine();
					}
				}
				str[i]=builder.toString().replaceAll("[^a-zA-Z]"," ");
				str[i]=str[i].replaceAll("\\s+", " ");
				TreeSet<String> spamSet=new TreeSet<String>();
				String[] temp=str[i].split(" ");
				for (int j = 0; j < temp.length; j++) {
					boolean b=spamSet.add(temp[j]);
					if(b==true||b==false){
						textc.append(temp[j]+" ");
					}
				}
				System.out.println("Spamset count:"+spamSet.size());
			}
			Iterator<String> iterator=V.iterator();
			Integer[] count=new Integer[V.size()];
			int j=-1;
			while (iterator.hasNext()) {
				j++;
				String itrText=iterator.next();
				vList.add(itrText);				
				count[j]=textc.toString().split(itrText).length-1;
			}
			iterator=V.iterator();
			j=0;
			int sum=0;
			for (int j2 = 0; j2 < count.length; j2++) {
				sum+=count[j2];
			}
			sum=sum+1;
			while (iterator.hasNext()) {
				condprob[j][i]= ((double)(count[j]+1)/(double)(sum+count.length));
				iterator.next();
				j++;
			}

		}
		return condprob;
	}
	private static ArrayList<String[]> extractVocab() throws Exception{
		File hamfolder = new File(hamTrainFile);
		File[] hamlistOfFiles = hamfolder.listFiles();
		StringBuilder builder=new StringBuilder(); 
		String[] returningObj=new String[hamlistOfFiles.length];
		for (int i = 0; i < hamlistOfFiles.length; i++) {
			builder=new StringBuilder();
			FileInputStream en=new FileInputStream(new File(hamlistOfFiles[i].toString()));
			BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(hamlistOfFiles[i].toString()))));
			String str=x.readLine();
			while (str!=null) {
				builder.append(str);
				str=x.readLine();
			}
			returningObj[i]=builder.toString();
		}

		File spamfolder = new File(spamTrainFile);
		File[] spamlistOfFiles = spamfolder.listFiles();
		String[] returningObj1=new String[spamlistOfFiles.length];
		for (int i = 0; i < spamlistOfFiles.length; i++) {
			builder=new StringBuilder();
			FileInputStream en=new FileInputStream(new File(spamlistOfFiles[i].toString()));
			BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(spamlistOfFiles[i].toString()))));
			String str=x.readLine();
			while (str!=null) {
				builder.append(str);
				str=x.readLine();
			}
			returningObj1[i]=builder.toString();
		}


		ArrayList<String[]> ret=new ArrayList<String[]>();
		ret.add(returningObj);
		ret.add(returningObj1);
		return ret;
	}

}
