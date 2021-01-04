# Area metric code library
This repo gathers code for computing the stochastic area metric. In Python and Matlab.

For more about the *stochastic area distance* the reader is referred to the following references:

[1] https://en.wikipedia.org/wiki/Wasserstein_metric
[2] Ramdas, Garcia, Cuturi “On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests” (2015). arXiv:1509.02237. https://arxiv.org/pdf/1509.02237.pdf


## Get started

Clone our repository using: 
``` bash
$ git@github.com:marcodeangelis/Area-metric.git
```

On MacOS/Linux:

``` bash
$ python3 -m venv venv 

$ source venv/bin/activate

(venv) $ pip install -r requirements.txt
```

On Windows10:

``` bash
$ python -m venv venv

$ venv\Scripts\activate

(venv) $ pip install -r requirements.txt
```

## Let's import the Python module for the *area metric*


```python
import areametric as am
```

### We can use the dot notation to access the following:

* The class `Dataset`
* The function `areaMe`
* The function `plot`

This is all we are going to need to demonstrate the use of the code library in this repository.

# Datasets of the same size
### Let's create two datasets with the same number of elements

Notice:

The datasets need to be created as Python `lists` and not as numpy arrays.


```python
d1 = [1,2,3,4,5,6,7,8]
d2 = [4.5]*8
```

We can already use the code to compute the area metric in just one line:


```python
print(am.areaMe(d1,d2))
```

    2.0


We can then use the `plot` function to visualise the results in a single plot.


```python
am.plot(d1,d2)
```


![png](fig/output_9_0.png)


Let's consider two different datasets.


```python
d1=5*[1.]
d2=5*[5.]
```

We would expect this difference to be equal to `4.0` as these two datasets are equvalent to two points.


```python
am.areaMe(d1,d2)
```




    4.0



The plot confirms this intuition.


```python
am.plot(d1,d2)
```


![png](fig/output_15_0.png)


## The `Dataset` class

We can make use of the `Dataset` class in several ways. One of these is to obtain the same results with some syntactic sugar, so in a more elegant way.


```python
d1,d2 = [1,2,3,4,5,6,7,8],[4.5]*8
D1 = am.Dataset(d1)
D2 = am.Dataset(d2)
```

We can plot the individual datasets:


```python
D1.plot()
```


![png](fig/output_19_0.png)


And we can compute the area metric using the subtraction operator `-`.


```python
D1-D2
```




    2.0



The infix operator `-` that implements the `area-metric` is commutative as we expected because it is in all respects a true metric. So swapping the operands will yield the same answer.


```python
D2-D1
```




    2.0



# Datasets of different size

This category of datasets will come soon.


```python

```
