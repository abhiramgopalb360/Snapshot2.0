# Snapshot2.0

Overview

Input

Non – Appended Data (  Raw client data )
<li>Either can use client-side attributes or append Data with Blend's inhouse 3rd party data ( Epsilon, Acxiom, Altair )
<li>Combined Dataset with Category column on which you want to compare populations\
<li>Define the Baseline ( Optional : Add/remove variables, set bins for numerical variables )

Output ( 3 Reports )
<li>Category Report : The base file to generate this category report comes from epsilon_attributes_binning.csv. ( More like tuning file ). This provides output in separate sheets based on category
<li>All variable Report : Contains all the variables of Data report in a single sheet with their respective indexes.
<li>PPT Report ( In excel format ) : Generates each variable in every sheet of excel & also includes in charts for each variable.

---
Best Practices
---
<li>If appended Data, it is common to filter it to 'Household' & 'Individual' level match
<li>Apply Regional/state filter as per the client's customer data.
<li>Sample size(3rd party Random Sample) should be same as category size
<li>Use the Epsilon_attributes_binning.xlsx to set your own Bins, Categories & Title for the Category Report
<li>Use EpsilonDataDictionary.xlsx to update definitions.
<li>Currently tested & working for 1 baseline + 2 categories ( max ); if more than 2 need to run the report twice.
