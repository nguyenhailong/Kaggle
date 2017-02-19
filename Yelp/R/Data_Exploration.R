# This script will copy a couple example photos from the input directory to the output directory

library(readr)

train_photo_to_biz_ids <- read_csv("../input/train_photo_to_biz_ids.csv")

for (i in 1:4) {
  cmd <- paste0("cp ../input/train_photos/", train_photo_to_biz_ids$photo_id[i], ".jpg .")
  cat("> ", cmd, "\n")
  system(cmd)
}