importClass(Packages.java.io.File)

function bleachCorrectStack(imgStack, startBleach, endBleach, frames, roiname)
{
	var frames = imgStack.getSize();
	var timeTrace = new java.lang.reflect.Array.newInstance(java.lang.Double.TYPE, frames);
	var counter = 0;
	var width = imgStack.getWidth();
	var height = imgStack.getHeight();
	/*MONO EXP*/
	var y0 = new Array(width * height);
	var A = new Array(width * height);
	var tau = new Array(width * height);
	
	/*DBLEXP*/
	/*var y0 = new Array(width * height);
	var A1 = new Array(width * height);
	var tau1 = new Array(width * height);
	var A2 = new Array(width * height);
	var tau2 = new Array(width * height);*(
	//var ratio = new Array(frames);
	/*for (var i = 0; i < frames; i++)
	{
		ratio[i] = new Array(width * height);
	}*/

	var y0Temp = new Array();
	var tauTemp = new Array();
	var ATemp = new Array();
	var bleachRate = new Array(width * height)  

	for (var i = 0; i < width; i++)
	{
		//IJ.log(i);
		for (var j = 0; j < height; j++)
		{
			var array = new java.lang.reflect.Array.newInstance(java.lang.Double.TYPE, endBleach-startBleach);
			//var arra
			for (var t = 0; t < (endBleach-startBleach); t++)
			{
				var pix = imgStack.getPixels(startBleach+t+1);
				array[t] = pix[i+ j*width]
			}

			var fitResults = doExpFit(array, false, 0.1);
			//fitResults = doDblExpFit(array, [fitResults.A, -1/fitResults.tau, fitResults.A, -1/fitResults.tau, fitResults.y0]);

			
			/*Mono Exp*/
			y0[i+ j*width] = fitResults.y0;
			A[i+ j*width] = fitResults.A;
			tau[i + j*width] = fitResults.tau;

			
			bleachRate[i+ j*width] = fitResults.bleachRate //0.00389863;

			counter++;		
		}
	}
	
	var fp1 = FloatProcessor(width, height, y0, null)  
	var fp2 = FloatProcessor(width, height, A, null)  
	var fp3 = FloatProcessor(width, height, tau, null)  
	var fp4 = FloatProcessor(width, height, bleachRate, null)
	
	var RK = new RankFilters();		//Median Filter, radius 10
	RK.rank(fp1, 10, 4);
	RK.rank(fp2, 10, 4);
	RK.rank(fp3, 10, 4);
	RK.rank(fp4, 10, 4);

	var y02 = fp1.getPixels();
	var A2 = fp2.getPixels();
	var tau2 = fp3.getPixels();
	var bleachRate2 = fp4.getPixels();
	/*
	var bcStack = new ImageStack(width, height);
	var ic = new ImageCalculator();
	for (var time = 0; time < frames; time+=1)
     	{
     		var ratio = new FloatProcessor(width, height, arrayratio(evaluateExp(A2, tau2, y02, 0.0, width, height), evaluateExp(A2, tau2, y02, (time + 1), width, height), width, height), null);

		var fp=imgStack.getProcessor(time+1);
		var tempFp = new ImagePlus("temp", fp);
		var tempRatio = new ImagePlus("ratio", ratio);
		var multiply = ic.run("Multiply create 32-bit (float)", tempFp, tempRatio);
		bcStack.addSlice("frame"+time+"bleachCorr", multiply.getProcessor());
		var stats = fp.getStatistics();
		timeTrace[time] = parseFloat(stats.mean);
	}*/
		var bcStack = new ImageStack(width, height)
	
	//var fp1 = FloatProcessor(width, height, bleachRate, null)    
	//var fp3 = FloatProcessor(width, height, y0, null)  
	//fp3.smooth();
	//fp3.blurGaussian(2);
	
	//y0Imp.show();
	var RK = new RankFilters();
	RK.rank(fp1, 5, 4);
	RK.rank(fp3, 5, 4);
	var bleachImp = ImagePlus("bleachRate", fp4)
	var y0Imp = ImagePlus("y0", fp1) 
	var ic = new ImageCalculator();
	
	for (var i = 0; i < frames; i++) 
	{
		var deltaBleachIP = new FloatProcessor(width, height);
		var deltaBleachImp = new ImagePlus("deltaBleach", deltaBleachIP);
		for (j = 0; j < i; j++)
		{
			var tempIP = imgStack.getProcessor(j+1);
			var tempImp = new ImagePlus("temp", tempIP);											
							
			var diff = ic.run("Subtract create 32-bit (float)", tempImp, y0Imp);
			var diffIP = diff.getProcessor();
			var diffMultiply = ic.run("Multiply create 32-bit (float)", diff, bleachImp);
			deltaBleachImp = ic.run("Add create 32-bit (float)", deltaBleachImp, diffMultiply);
			deltaBleachImp.updateImage()
		}
		var oldSliceImp = new ImagePlus("newSlice", imgStack.getProcessor(i+1));
		var newSliceImp = ic.run("Add create 32-bit (float)", deltaBleachImp, oldSliceImp);
		var newSliceIP = newSliceImp.getProcessor();
		bcStack.addSlice(roiname+"_bleachCorrected_"+i+"", newSliceIP);		
		var stats = newSliceIP.getStatistics();
		timeTrace[i] = parseFloat(stats.mean);
	}
		
		return  {
			'bcStack': bcStack,
			'timeTrace': timeTrace
		};
}

function min(array)
{
	var min = array[0];	
	for(var i = 0; i < array.length; i++)
	{
		if ( array[i] < min )
		{
			min = array[i];
		}
	}
	return min;
}

function max(array)
{
	var max = array[0];	
	for(var i = 0; i < array.length; i++)
	{
		if ( array[i] > max )
		{
			max = array[i];
		}
	}
	return max;
}
 
Array.prototype.writeIndices = function( n ) {
    for( var i = 0; i < (n || this.length); ++i ) this[i] = i;
    return this;
}

function median(array) 
{
	array.sort(function(a,b){return a - b})
    	var middle = Math.floor(array.length/2);
    	if (array.length%2 == 1) 
    	{
        	return array[middle];
        }
        else
        {
        	return (array[middle-1] + array[middle]) / 2.0;
    	}
}

function bleachCorrectArray(array, bleachRate, y0)
{
	var output = new java.lang.reflect.Array.newInstance(java.lang.Double.TYPE, array.length);
	
	for(var i = 0; i < array.length; i+=1)
	{
		var deltaBleach = 0;
		for (var j = 0; j < i; j+=1)
		{
			deltaBleach += (bleachRate * (array[j] - y0) );
		}

		output[i] =  array[i] + deltaBleach;
	}

	return output;
	
}
function normalize(array)
{
	var output = new java.lang.reflect.Array.newInstance(java.lang.Double.TYPE, array.length);
	
	var max = array[0];
	var min = array[0];
	
	for(var i = 0; i < array.length; i+=1)
	{
		if ( array[i] > max )
		{
			max = array[i];
		}
		if ( array[i] < min )
		{
			min = array[i];
		}
	}
	for(var i = 0; i < array.length; i+=1)	output[i] = (array[i] - min)/(max-min);	
	
	return output;

}


function Matrix3x3Inverse(A)
{
	var m = A.length;
	var n = A[0].length;
	
	var a = A[0][0]
	var b = A[0][1]
	var c = A[0][2]
	
	var d = A[1][0]
	var e = A[1][1]
	var f = A[1][2]

	var g = A[2][0]
	var h = A[2][1]
	var k = A[2][2]
	
	var detA = a*(e*k-f*h)-b*(k*d-f*g)+c*(d*h-e*g);
	var invA = new Array(m);
	for (var i = 0; i < m; i++)
	{
		invA[i] = new Array(n);
		for (var j = 0; j < n; j++) 
		{
    			invA[i][j] = 0;
  		}
	}	
	invA[0][0] = (e*k-f*h)/detA
	invA[0][1] = -(b*k-c*h)/detA
	invA[0][2] = (b*f-c*e)/detA
	
	invA[1][0] = -(d*k-f*g)/detA
	invA[1][1] = (a*k-c*g)/detA
	invA[1][2] = -(a*f-c*d)/detA
	
	invA[2][0] = (d*h-e*g)/detA
	invA[2][1] = -(a*h-b*g)/detA
	invA[2][2] = (a*e-b*d)/detA

	return invA
}
function MatrixMult(A, B)//a[m][n], b[n][p]
{
	var n = A[0].length;
	var m = A.length;
	var p = B[0].length;
		
	var ans  = [[,],[,]];

	var ans = new Array(m);
	for (var i = 0; i < m; i++) 
	{
  		ans[i] = new Array(p);
  		for (var j = 0; j < p; j++) 
  		{
    			ans[i][j] = 0;
  		}
	}
	for(var i = 0;i < m;i++)
		for(var j = 0;j < p;j++)
			for(var k = 0;k < n;k++)
				ans[i][j] += A[i][k] * B[k][j];

   	return ans;
}

function guessP(y)
{
	var n = y.length
	var Yoi = new Array(n)
	
	var B11=0, B12=0, B13=0, B21=0, B22=0, B23=0, B31=0, B32=0, B33=0
	var C1=0, C2=0, C3=0
	var sumyoi = 0
	for(var i = 0; i < n; i+=1)
	{
		sumyoi+=y[i]
		Yoi[i] = sumyoi
	}	
	for(var i = 0; i < n; i+=1)
	{
		B11+=(Yoi[i]*Yoi[i])
		B12+=((i+1)*Yoi[i])
		B13+=Yoi[i]
		B21+=((i+1)*Yoi[i])
		B22+=((i+1)*(i+1))
		B23+=(i+1)
		B31=B13
		B32=B23
		B33=n
		C1 += ( Yoi[i]*y[i] )
		C2 += ( (i+1) * y[i] )
		C3 += (y[i] )
	}
	
	var matA=[[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
	var matB= [[C1], [C2], [C3]]
	var invA = Matrix3x3Inverse(matA)
	var result = MatrixMult(invA, matB)
	
	return 1/(1-result[0][0])	
}

function sumpi(p, n)
{
	return (p/(1-p)) * (1-Math.pow(p, n))
}
function sumipi(p, n)
{
	return ( p / Math.pow( 1-p, 2) ) * ( 1- ( n * ( 1-p) +1) * Math.pow(p, n))
}
function sumi2pi(p, n)
{
	return ( p / Math.pow(1-p, 3) ) * ( 1 + p - ( Math.pow( n*(1-p)+1 ,2) + p ) * Math.pow(p,n))
}

function dF1dp(p, y)
{
	var n = y.length, i = 0;
	var sum1 = 0, sum2 = 0
	for ( i = 0; i < n ; i += 1)
	{
		sum1+= ( y[i]*(i+1) * Math.pow(p, i+1))
		sum2+= y[i]
	}
	return ( 1/p ) * ( n*sum1 - sum2*sumipi(p, n) )
}

function dF2dp(p, y)
{
	var n = y.length	
	return (2/p) * (n + sumipi(Math.pow(p, 2), n) - sumpi(p, n)*sumipi(p, n) )
}

function dF3dp(p, y)
{
	var n = y.length, i = 0
	var sum1=0, sum2 =0, sum3=0
	for ( i = 0; i < n ; i += 1)
	{
		sum1+= y[i]
		sum2+= (y[i]*Math.pow(p, i+1))
		sum3+=(y[i]*(i+1)*Math.pow(p, i+1))
	}	
	return (1/p) * ( 2* sum1*sumipi(Math.pow(p, 2), n)*n*sumipi(Math.pow(p, 2), n)-sumipi(p, n) * sum2 - sumpi(p, n)*sum3)
}

function dadp(p, y)
{
	return (F2(p, y) * dF2dp(p, y) - F1(p, y)*dF2dp(p,y)) / Math.pow(F2(p, y), 2)
}

function dcdp(p, y)
{
	return (F2(p, y) * dF3dp(p, y) - F3(p, y)*dF2dp(p,y)) / Math.pow(F2(p, y), 2)
}

function F1(p, y)
{
	var n = y.length, i = 0
	var sum1=0, sum2 =0
	for ( i = 0; i < n ; i += 1)
	{
		sum1+= ( y[i] * Math.pow(p, i+1))
		sum2+= y[i]
	}	
	return n*sum1-sum2*sumpi(p, n)
}

function F2(p, y)
{
	var n = y.length
	return n * sumpi(Math.pow(p, 2), n) - Math.pow(sumpi(p, n), 2) 
}

function F3(p, y)
{
	var n = y.length, i = 0
	var sum1=0, sum2 =0
	for ( i = 0; i < n ; i += 1)
	{
		sum1+= y[i]
		sum2+= ( y[i] * Math.pow(p, i+1))
	}
	
	return sum1*sumpi(Math.pow(p, 2), n)-sum2*sumpi(p, n)
}

function F(p, y)
{
	var n = y.length, i=0, sum1=0
	for ( i = 0; i < n ; i += 1)
		sum1+= ( (i+1) * y[i] * Math.pow(p, i+1) )
	
	return (F1(p,y)/F2(p,y))*sumipi(Math.pow(p, 2), n) + (F3(p,y)/F2(p,y))*sumipi(p, n) - sum1
}
function dFdp(p, y)
{
	var n = y.length, i = 0
	var sum1=0, sum2 =0
	for ( i = 0; i < n ; i += 1)
		sum1+= (y[i] * Math.pow(i+1, 2) * Math.pow(p, i+1) )
	return dadp(p, y)*sumipi(Math.pow(p, 2), n)+dcdp(p, y)*sumipi(p, n)+(2*(F1(p,y)/F2(p,y)) / p)* sumi2pi(Math.pow(p, 2), n)+ ((F3(p,y)/F2(p,y)) / p) * sumi2pi(p, n) - (1/p) * sum1
}
function doExpFit(y)
{
	var xaxis = [].writeIndices(y.length);
	var fitModelOutput = new java.lang.reflect.Array.newInstance(java.lang.Double.TYPE, xaxis.length);	
	var p = guessP(y)
//	IJ.log("guessed p: " + p);
	var pnew = 1
	var e = 0.001
	while(1)
	{
		if( Math.log(pnew/p) > e)
		{	
			pnew = p - F(p, y) / dFdp(p, y)
			p = pnew
		}
		else{
			for( var i = 0; i < xaxis.length; i+=1)
				fitModelOutput[i] = ( F1(p, y) / F2(p, y) )*Math.exp( Math.log(p)*(xaxis[i]) ) + ( F3(p, y) / F2(p, y) );

			return {
        			'A': ( F1(p, y) / F2(p, y) ),
        			'y0': ( F3(p, y) / F2(p, y) ),
        			'tau': (-1/Math.log(p)),
        			'ModelArray': fitModelOutput,
        			'bleachRate':(1-p),
        			'result': "A = " + ( F1(p, y) / F2(p, y) ) + " ; tau = " + (-1/Math.log(p)) + " ; y0 = " +( F3(p, y) / F2(p, y) ) +""
        		}; 
		}
	}	
}

function doExpFit(y, calculateModelArray, e)
{
	var xaxis = [].writeIndices(y.length);
	var fitModelOutput = new java.lang.reflect.Array.newInstance(java.lang.Double.TYPE, xaxis.length);	
	var p = guessP(y)

	var pnew = 1
	//var e = 0.001
	while(1)
	{
		if( Math.log(pnew/p) > e)
		{	
			pnew = p - F(p, y) / dFdp(p, y)
			p = pnew
		}
		else{
			if(calculateModelArray)
			{
				for( var i = 0; i < xaxis.length; i+=1)
					fitModelOutput[i] = ( F1(p, y) / F2(p, y) )*Math.exp( Math.log(p)*(xaxis[i]) ) + ( F3(p, y) / F2(p, y) );
			}

			return {
        			'A': ( F1(p, y) / F2(p, y) ),
        			'y0': ( F3(p, y) / F2(p, y) ),
        			'tau': (-1/Math.log(p)),
        			'ModelArray': fitModelOutput,
        			'bleachRate':(1-p),
        			'result': "A = " + ( F1(p, y) / F2(p, y) ) + " ; tau = " + (-1/Math.log(p)) + " ; y0 = " +( F3(p, y) / F2(p, y) ) +""
        		}; 
		}
	}	
}




