# Local datasets

Put datasets here on your AutoDL server; images/masks are excluded from Git.
Example root: data/VegAnn with JPEGImages/, SegmentationClass/ and
ImageSets/Segmentation/{train,val,test}.txt. Use --data-path data/VegAnn.
The current loader requires 512x512 images and masks, with mask values
0=background and 255=plant (not 0/1). Do not train on visualization colors.
Split lists use ImageSets/Segmentation, not ImageSets/Main.
