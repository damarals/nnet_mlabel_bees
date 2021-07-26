library(imager)
library(tidyverse)

# Convolução
padding_image <- function(img, ncells) {
  img_new <- matrix(0, nrow = nrow(img) + 2*ncells, 
                    ncol = ncol(img) + 2*ncells)
  img_new[ncells + 1:nrow(img), ncells + 1:ncol(img)] <- img
  
  return(img_new)
}
apply_linear_kernel <- function(img, kernel) {
  pad_size <- (nrow(kernel) - 1)/2
  img <- padding_image(img = img, ncells = pad_size)
  
  img_new <- outer(
    X = (1 + pad_size):(nrow(img) - pad_size),
    Y = (1 + pad_size):(ncol(img) - pad_size),
    
    FUN = Vectorize(function(rw, cl) {
      img_subset <- img[(rw - pad_size):(rw + pad_size),
                        (cl - pad_size):(cl + pad_size)]
      
      result <- sum(img_subset * kernel)
      
      return(result)
    }))
  
  return(img_new)
}
# Filtro Laplaciano
kernel_laplacian <- function(diagonals = F) {
  if(diagonals) return(-1*matrix(c(1, 1, 1, 1, -8, 1, 1, 1, 1), 3)) 
  else return(-1*matrix(c(0, 1, 0, 1, -4, 1, 0, 1, 0), 3)) 
}

# Leitura das Imagens
paths <- fs::dir_ls('data/bee_imgs/')
image_list <- load.dir(paths)

# Variância do Laplaciano
var_laplacian <- lapply(image_list, function(img) {
  if(dim(img)[4] > 3) return(NULL)
  img %>%
    grayscale %>%
    apply_linear_kernel(., kernel_laplacian()) %>%
    c %>% var
}) %>% do.call(rbind, .) %>% 
  as.data.frame %>% rownames_to_column