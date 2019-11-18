#' Changing the names of the hyperparameters (HP)
#' for plotting purposes
#' @param data DataFrame containing the HP names 
#' and their variance contributions
#' @param algorithm str specifing the machine learning algorithm
#' takes one of the following values 
#' {'RandomForest','ExtraTrees','AdaBoost'}
#' @return  character column containing the modified HP names


param_names <- function(data, algorithm){
  
  require(gsubfn)
  
  
  data$param <- gsubfn(".",
                       list("," = " /",
                            "(" = "",
                            ")" = "",
                            "'" = ""),
                       data$param)
  
  if (is.finite(match(algorithm, c('RandomForest',
                         'ExtraTrees',
                         'DecisionTree')))){
    
    data$param <- gsubfn('.',
                         list("criterion" = "split criterion",
                              "min_samples_split" = "min. samples split",
                              "max_features" = "max. features",
                              "min_samples_leaf" = "min. samples leaf"),
                         data$param)


  }
  else if(algorithm == 'AdaBoost'){
    
    data$param <- gsubfn('.',
                         list("n_estimators" = "iterations",
                              "learning_rate" = "learning rate"),
                         data$param)
  }
  

  return(data$param)
}

#' Creates violin plots of the hyperparameter (HP)
#' variance contributions
#' 
#' @param data DataFrame containing the HP names 
#' and their variance contributions
#' @param top_n integer value specifying the number
#' of most important HP combinations sorted in descending order of
#' their mode values, if NA then plots all HP combinations 
#' @param horizontal boolean plots the figures horizontally 
#' in case of TRUE 
#' @return  character column containing the modified HP names



violin_plot <- function(data, top_n=NA, horizontal=T){
  
  require(dplyr)
  require(ggplot2)
  
  top <- data %>%
    group_by(param)%>% 
    summarise(median=median(importance))%>% 
    arrange(desc(median))
  
  
  if (is.na(top_n)){
    top_params <- top$param
  } else {
    top_params <- top$param[1:top_n]
  }
  
  
  selected <- is.finite(match(data$param,top_params))
  
  dat_selected <- data[selected,]
  
  p <- ggplot(dat_selected, 
              aes(x=reorder(param,
                            importance,
                            FUN = median),
                  y=importance,
                  fill=param)
  ) + 
    geom_violin(scale = 'width') +
    
    geom_boxplot(width=0.15, fill = 'white') +
    theme(axis.text.x = element_text(size = 15),
          axis.text.y = element_text(size = 15),
          axis.title.x = element_text(size = 15), 
          legend.position = 'none') +
    labs(x = '',
         y = 'Variance Contribution')
  
  if (horizontal){
    p + coord_flip()  
  } else {
    p
  }
  
}



# importing the data

f_data <- read.csv('DecisionTree_fANOVA_results.csv', 
                 colClasses = c("character", "numeric", rep("NULL", 3)))
head(f_data)

#                         param  importance      
# 1                      algorithm 0.001831505
# 2                  learning_rate 0.049146259
# 3                      max_depth 0.439507044
# 4                   n_estimators 0.028026811
# 5 ('learning_rate', 'max_depth') 0.114227831
# 6  ('max_depth', 'n_estimators') 0.090332874

# filter only imputation case
f_data <- read.csv('DecisionTree_fANOVA_results.csv') 
f_data <- f_data[f_data$imputation == 'True',1:2]
                   
head(f_data)
f_data$param <- as.character(f_data$param)
f_data$param <- param_names(f_data,'DecisionTree')

violin_plot(f_data,
            top_n = 10,
            horizontal = T)

