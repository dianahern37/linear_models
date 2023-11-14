Cross Validation
================
Diana Hernandez
2023-11-14

Load libraries and set seed for reproducibility.

``` r
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.3     ✔ readr     2.1.4
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.0
    ## ✔ ggplot2   3.4.3     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.2     ✔ tidyr     1.3.0
    ## ✔ purrr     1.0.2     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(modelr)
library(mgcv)
```

    ## Loading required package: nlme
    ## 
    ## Attaching package: 'nlme'
    ## 
    ## The following object is masked from 'package:dplyr':
    ## 
    ##     collapse
    ## 
    ## This is mgcv 1.9-0. For overview type 'help("mgcv-package")'.

``` r
set.seed(1)
```

# Nonlinear data and CV

``` r
nonlin_df = 
  tibble(
    id = 1:100,
    x = runif(100, 0, 1),
    y = 1 - 10 * (x - .3) ^ 2 + rnorm(100, 0, .3)
  )

nonlin_df |> 
  ggplot(aes(x = x, y = y)) + 
  geom_point()
```

![](cross_validation_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

Do the train / test split.

``` r
train_df = sample_n(nonlin_df, 80)
test_df = anti_join(nonlin_df, train_df, by = "id")

ggplot(train_df, aes(x = x, y = y)) + 
  geom_point() + 
  geom_point(data = test_df, color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Models

``` r
linear_model = lm(y ~ x, data = train_df)
smooth_model = mgcv::gam(y ~ s(x), data = train_df)
wiggly_model = mgcv::gam(y ~ s(x, k = 30), sp = 10e-6, data = train_df)
```

Quick visualization!

``` r
train_df |> 
  modelr::add_predictions(smooth_model) |> 
  ggplot(aes(x = x, y = y)) + 
  geom_point() + 
  geom_line(aes(y = pred))
```

![](cross_validation_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
train_df |> 
  modelr::add_predictions(wiggly_model) |> 
  ggplot(aes(x = x, y = y)) + geom_point() + 
  geom_line(aes(y = pred), color = "red")
```

![](cross_validation_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Root mean squared errors (RMSEs) for each model.

``` r
rmse(linear_model, train_df)
```

    ## [1] 0.7178747

``` r
rmse(smooth_model, train_df)
```

    ## [1] 0.2874834

``` r
rmse(wiggly_model, train_df)
```

    ## [1] 0.2498309

``` r
rmse(linear_model, test_df)
```

    ## [1] 0.7052956

``` r
rmse(smooth_model, test_df)
```

    ## [1] 0.2221774

``` r
rmse(wiggly_model, test_df)
```

    ## [1] 0.289051
