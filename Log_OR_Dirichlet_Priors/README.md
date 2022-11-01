## Log Odds Ratios with Dirichlet Priors

The Create_Word_Lists.ipynb creates lists of all medical and nonmedical words that occur in patient notes, stratified by patient race. For example, if the word "intestinal" occurred 1408 times across all White patient notes, that word will also appear 1408 times in the list of medical words for White patients. These lists of words are used to compute the log odds ratios in Log_OR_Dirichlet_Prior_Race.ipynb


Log_OR_Dirichlet_Prior_Race.ipynb computes the log odds ratios to compare word frequency across patient race. It also computes the associated z-scores and confidence intervals. Non-hispanic White patient notes are compared to Black patient notes, and all other notes are compared to White patient notes.
