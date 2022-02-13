library(quantreg)
library(ggpubr)
library(ggplot2)
library(dplyr)
library(tidyr)

n < - 100
p < - 1
nout < - 10
x < - matrix(rnorm(n * p, -5, 5), nc=p)
x.data < - cbind(rep(1, n), x)
beta < - matrix(c(0, rep(2, p)))
y < - drop(x.data % * % beta) + rnorm(n)
out.idx < - sample(1: n, nout)
y[out.idx] < - y[out.idx] + 10
data < - data.frame(y, x)
real.outliner < - data[out.idx,]

res.lm < - lm(y
~ x, data = data)
ols.result < - res.lm$coefficients

## make variable and first yadj
## for calculate varibale need to use rq()
model.lad < - rq(y
~x, tau = 0.5, data = data.frame(x=x, y=y))
beta.old < - model.lad$coefficients
variable < - median(abs((model.lad$residuals) ^ 2))
x.hat < - as.matrix(x.data) % * % solve(t( as.matrix(x.data)) % * % as.matrix(x.data)) % * % t( as.matrix(
    x.data))  ##x hat matrix
y.predict < - x.data % * % beta.old  ##calculate yhat use x * beta hat
smallr.old < - y - y.predict  ##the first smallr

##thresholding function

scad < -function(x){ 
    ifelse(abs(x)<=lambda,return(0),return(x-sign(x)*lambda))}
  

###function which use for loop
lambda.choose < - function( lambda, method){
b < -0
continuechange=TRUE
while (continuechange){
jadj < - y - smallr.old  ## calculate adj y
beta.new < - solve(t(x.data) % * % x.data) % * % t(x.data) % * % jadj  ##calculate new beta
y.hat < - as.matrix(x.data) % * % beta.new  ##Predictive value y
r < - y-y.hat  ##calculate R
smallr.new < - apply(r, 1, FUN=method)  ##use threholding function to calculate new smallr
smallr.new < - matrix(smallr.new)
changer < - max(abs(smallr.old-smallr.new))
smallr.old < - smallr.new
beta.old < - beta.new
continuechange=ifelse(changer <= 0.0001, F, T)
b < -b+1
if (b >= 1000){
break
}
}
outerliner.position < - which(smallr.old != 0)
# find position of point which is outliners
number.outliner < - length(outerliner.position)  ## number of outliners
m = n - p
k = number.outliner + 1
rss < - sum(((diag(n) - x.hat) % * % (as.matrix(y) - smallr.old)) ^ 2)
bic < - n * log(variable) + n * log(2 * pi) + rss / variable + log(n) * k
return (c(number.outliner = number.outliner, bic = bic, lambda = lambda, beta.old = beta.old, smallr.old = smallr.old))
}






###make the result to plot( lambda and beta path and real plot)
result.plot < - function(dataset, variable.show = 1){
beta.bound < - 4 + p
bestmodel.beta < - dataset[which.min(dataset[, 2]), ][4: beta.bound]
bestmodel.
lambda < - dataset[which.min(dataset[, 2]), ][3]
sample.beta.path < - dataset[, c(4:beta.bound)]
sample.beta.path < - cbind(lambda .grid, sample.beta.path)
df.beta.path < - as.data.frame(sample.beta.path) % > % pivot_longer(col=-
lambda .grid, names_to="beta", values_to="estimate.beta") % > % arrange(lambda .grid)
                                                                               beta.path.plot < - ggplot(df.beta.path,
                                                                               aes(x
                                                                                   = lambda.grid, y = estimate.beta, col = beta)) + geom_line() + guides(
    color="none") + geom_vline(aes(xintercept=bestmodel.
lambda ), colour="#BB0000", linetype="dashed")
path.bound < - p + 5
sample.point.path < - dataset[, c(path.bound:ncol(dataset))]
sample.point.path < - cbind(lambda .grid, sample.point.path)

df.smallr.path < - as.data.frame(sample.point.path) % > % pivot_longer(col=-
lambda .grid, names_to="gamma", values_to="estimate.gamma") % > % arrange(lambda .grid)
                                                                                 smallr.path.plot < - ggplot(df.smallr.path,
                                                                                 aes(x
                                                                                     = lambda.grid, y = estimate.gamma, col = gamma)) + geom_line() + guides(
    color="none") + geom_vline(aes(xintercept=bestmodel.
lambda ), colour="#BB0000", linetype="dashed")
final.result.gamma < - df.smallr.path$estimate.gamma[which(df.smallr.path$lambda .grid == bestmodel.lambda )]
                                                                                 outerliner.position < - which(final.result.gamma != 0)
                                                                                 number.outliner.detest < - length(outerliner.position)
                                                                                 s < - (number.outliner.detest - length(intersect(outerliner.position,
                                                                                 out.idx))) / (n - nout)
m < - 1 - (length(intersect(outerliner.position, out.idx))) / nout
jd < - (length(intersect(outerliner.position, out.idx))) / nout
outliner < - data[outerliner.position,]
plot.data < - data[, c(1, 1 + variable.show)]
colnames(plot.data) < - c("y", "x")
plot.outliner.data < - outliner[, c(1, 1 + variable.show)]
colnames(plot.outliner.data) < - c("y", "x")
plot.real.outliner.data < - real.outliner[, c(1, 1 + variable.show)]
colnames(plot.real.outliner.data) < - c("y", "x")
true.plot < - ggplot(plot.data, aes(x, y)) + geom_point() + geom_abline(intercept=bestmodel.beta[1],
                                                                        slope=bestmodel.beta[1 + variable.show],
                                                                        col='pink', linetype="dashed") + geom_abline(
    intercept=0, slope=2, col='blue') + geom_point(plot.outliner.data, mapping=aes(x, y), col='green') + geom_point(
    plot.real.outliner.data, mapping=aes(x, y), col='blue', shape=21)
plo < - ggarrange(true.plot,
                  ggarrange(beta.path.plot, smallr.path.plot, ncol=1, nrow=2, labels=c("beta.path", "gamma.path")),
                  nrow=2, labels="result")
return (list(bestmodel.
        lambda = bestmodel.lambda, bestmodel.beta = bestmodel.beta, final.result.gamma = final.result.gamma, s=s, m = m, jd = jd, plo))
}


lambda .max < - max(abs(drop((diag(n)-x.hat) % * % y) / sqrt(1-diag(x.hat))))
       lambda.grid < - 2 ^ seq(-4, log2(lambda .max), length=50)

coeff.soft < - NULL
for (lambda in lambda.grid){
            coeff.soft < - rbind(coeff.soft, lambda.choose( lambda, scad))
}

data.draw.
lambda < - data.frame(bic.adj = coeff.soft[, 2], lam=lambda .grid)

                                                            ggplot()+
                                                            geom_line(data=data.draw.lambda, aes(y = bic.adj, x = lam))





result.plot(coeff.soft, variable.show = 1)
