function result = normalize_0_255(A)
ymin=0;
ymax=255;
xmin = min(min(A)); 
xmax = max(max(A)); 
result = round((ymax-ymin)*(A-xmin)/(xmax-xmin) + ymin); 
end