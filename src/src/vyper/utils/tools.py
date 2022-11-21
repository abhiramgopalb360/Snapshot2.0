from typing import Tuple

import pandas as pd
import numpy as np

import pandas.core.algorithms as algos
import warnings
import inspect


class DataTools:

    @staticmethod
    def isnumeric(vec):
        # TODO look into pd.to_numeric
        try:
            for i in range(len(vec)):
                if vec[i] is not None:
                    float(vec[i])
            return True
        except ValueError:
            return False

    @staticmethod
    def get_excluded_epsilon_columns() -> set:
        # TODO implement connector to Snowflake to pull Epsilon names
        # TODO or implement read from yml.
        known_epsilon_columns = {"file_code", "record_quality_code", "num_sourc_verify_hh", "fips_state_code", "zip",
                                 "zip4", "delivery_point_code", "carrier_route", "contracted_address",
                                 "post_office_name", "state", "county_code", "addr_quality_code", "addr_typ",
                                 "verification_date_hh", "surname", "agility_addr_key", "agility_hh_key",
                                 "person_seq_no1", "ttl_code1", "given_name1", "middle_initital_1", "gender1",
                                 "member_code_person1", "verification_date_person1", "african_american_conf_code1",
                                 "assimilation_code1", "ethnic_group1", "ethnic_group_code1", "hisp_country_origin1",
                                 "language_code1", "religion1", "num_tradelines1", "bankcard_issue_date1",
                                 "adv_ind_marital_status1", "adv_ind_marital_stats_indicatr1",
                                 "bthday_person_w_day_enh1", "bthday_mth_indicator_enh1", "individual_exact_age1",
                                 "self_reported_responder1", "agility_individual_key1", "person_seq_no2", "ttl_code2",
                                 "given_name2", "middle_initital_2", "gender2", "member_code_person2",
                                 "verification_date_person2", "african_american_conf_code2", "assimilation_code2",
                                 "ethnic_group2", "ethnic_group_code2", "hisp_country_origin2", "language_code2",
                                 "religion2", "num_tradelines2", "bankcard_issue_date2", "adv_ind_marital_status2",
                                 "adv_ind_marital_stats_indicatr2", "bthday_person_w_day_enh2",
                                 "bthday_mth_indicator_enh2", "individual_exact_age2", "self_reported_responder2",
                                 "agility_individual_key2", "person_seq_no3", "ttl_code3", "given_name3",
                                 "middle_initital_3", "gender3", "member_code_person3", "verification_date_person3",
                                 "african_american_conf_code3", "assimilation_code3", "ethnic_group3",
                                 "ethnic_group_code3", "hisp_country_origin3", "language_code3", "religion3",
                                 "num_tradelines3", "bankcard_issue_date3", "adv_ind_marital_status3",
                                 "adv_ind_marital_stats_indicatr3", "bthday_person_w_day_enh3",
                                 "bthday_mth_indicator_enh3", "individual_exact_age3", "self_reported_responder3",
                                 "agility_individual_key3", "person_seq_no4", "ttl_code4", "given_name4",
                                 "middle_initital_4", "gender4", "member_code_person4", "verification_date_person4",
                                 "african_american_conf_code4", "assimilation_code4", "ethnic_group4",
                                 "ethnic_group_code4", "hisp_country_origin4", "language_code4", "religion4",
                                 "num_tradelines4", "bankcard_issue_date4", "adv_ind_marital_status4",
                                 "adv_ind_marital_stats_indicatr4", "bthday_person_w_day_enh4",
                                 "bthday_mth_indicator_enh4", "individual_exact_age4", "self_reported_responder4",
                                 "agility_individual_key4", "person_seq_no5", "ttl_code5", "given_name5",
                                 "middle_initital_5", "gender5", "member_code_person5", "verification_date_person5",
                                 "african_american_conf_code5", "assimilation_code5", "ethnic_group5",
                                 "ethnic_group_code5", "hisp_country_origin5", "language_code5", "religion5",
                                 "num_tradelines5", "bankcard_issue_date5", "adv_ind_marital_status5",
                                 "adv_ind_marital_stats_indicatr5", "bthday_person_w_day_enh5",
                                 "bthday_mth_indicator_enh5", "individual_exact_age5", "self_reported_responder5",
                                 "agility_individual_key5", "file_id", "install_date", "vehicle_last_seen_date_1",
                                 "vehicle_last_seen_date_2", "vehicle_last_seen_date_3",
                                 "vehicle_last_seen_date_4", "vehicle_last_seen_date_5", "vehicle_last_seen_date_6",
                                 "vehicle_year_1", "vehicle_year_2", "vehicle_year_3", "vehicle_year_4",
                                 "vehicle_year_5", "vehicle_year_6"}

        return known_epsilon_columns

    def epsilon_batch_recoder(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, set]:
        # TODO catch up with Abid about this.
        # recode boolean
        variable_set_1 = {'OCREDIT_AUTO_LOANS', 'OCREDIT_EDU_STUDENT_LOANS', 'OCREDIT_FIN_SERVICES_BANKING',
                          'OCREDIT_FIN_SERVICES_INSTALL', 'OCREDIT_FIN_SERVICES_INSURANCE', 'OCREDIT_LEASING',
                          'OCREDIT_HOME_MORTG', 'CREDIT_ACTIVE', 'GOLD_WALLETS', 'SILVER_WALLETS', 'PLATINUM_WALLETS',
                          'BOOK_BEST_SELLING_FICTION_ALL', 'BOOK_BIBLE_DEVOTIONAL_ALL', 'BOOK_BOOKS_READING_ALL',
                          'BOOKS_SPORTS_ALL', 'BOOK_CHILDREN_BOOK_ALL', 'BOOK_COOKING_CULINARY_ALL',
                          'BOOK_COUNTRY_LIFESTYLE_ALL', 'BOOK_ENTERTAINMENT_ALL', 'BOOK_FASHION_ALL',
                          'BOOK_INTERIOR_DECORATING_ALL', 'BOOK_MEDICAL_OR_HEALTH_ALL', 'BOOK_MILITARY_ALL',
                          'BOOK_ROMANCE_ALL', 'BOOK_WORLD_NEWS_ALL', 'DONOR_DONATE_CHARIT_CAUSES_ALL',
                          'DONOR_ACTIVE_MILITARY_ALL', 'DONOR_ALZHEIMER_ALL', 'DONOR_ANIMAL_WELFARE_ALL',
                          'DONOR_ARTS_CULTURAL_ALL', 'DONOR_CANCER_ALL', 'DONOR_CATHOLIC_ALL', 'DONOR_CHILDREN_ALL',
                          'DONOR_HUMANITARIAN_ALL', 'DONOR_NATIVE_AMERICAN_ALL', 'DONOR_OTHER_RELIGIOUS_ALL',
                          'DONOR_POLITICAL_CONSERV_ALL', 'DONOR_POLITICAL_LIBERAL_ALL', 'DONOR_VETERAN_ALL',
                          'DONOR_WORLD_RELIEF_ALL', 'DONOR_WILDLIFE_ENVIRN_CAUS_ALL', 'COLLECT_ANY_ALL',
                          'COLLECT_ART_ANTIQUE_ALL', 'COLLECT_DOLLS_ALL', 'COLLECT_FIGURINES_ALL', 'COLLECT_STAMPS_ALL',
                          'COLLECT_COIN_ALL', 'HOBBIES_AUTOMOTIVE_WORK_ALL', 'HOBBIES_BAKING_ALL',
                          'HOBBIES_BIRD_FEED_WATCH_ALL', 'HOBBIES_CAREER_ADVC_COURSE_ALL', 'HOBBIES_CIGAR_SMOKING_ALL',
                          'HOBBIES_CONTEST_SWPSTAKES_ALL', 'HOBBIES_COOKING_ALL', 'HOBBIES_CRAFTS_ALL',
                          'HOBBIES_CULTURAL_EVENTS_ALL', 'HOBBIES_GARDENING_ALL', 'HOBBIES_GOURMET_FOODS_ALL',
                          'HOBBIES_ANY_ALL', 'HOBBIES_HOME_IMPROV_DIY_ALL', 'HOBBIES_HOME_STUDY_COURSE_ALL',
                          'HOBBIES_MOTORCYCLE_RIDING_ALL', 'HOBBIES_PHOTOGRAPHY_ALL', 'HOBBIES_QUILTING_ALL',
                          'HOBBIES_SCRAPBOOKING_ALL', 'HOBBIES_SELF_IMPROVE_COZS_ALL', 'HOBBIES_SEWING_KNITTING_ALL',
                          'HOBBIES_WINE_ALL', 'HOBBIES_WOODWORKING_ALL', 'INVEST_INSUR_BURIAL_INSUR_ALL',
                          'INVEST_INSUR_INSURANCE_ALL', 'INVEST_INSUR_INVESTMENTS_ALL',
                          'INVEST_INSUR_JUVLIFE_INSUR_ALL', 'INVEST_INSUR_LIFE_INSUR_ALL',
                          'INVEST_INSUR_MEDICARE_COV_ALL', 'INVEST_INSUR_MUTUAL_FUNDS_ALL', 'INVEST_INSUR_STOCKS_ALL',
                          'MO_ANY_ALL', 'MO_APPRL_ALL', 'MO_BIG_TALL_ALL', 'MO_BOOKS_ALL', 'MO_CHILDREN_PROD_ALL',
                          'MO_FOOD_ALL', 'MO_GIFTS_ALL', 'MO_HEALTH_BEAUTY_PROD_ALL', 'MO_HOME_FURNISHING_ALL',
                          'MO_JEWELRY_ALL', 'MO_MAGAZINES_ALL', 'MO_VIDEOS_DVD_ALL', 'MO_WOMEN_PLUS_ALL',
                          'MUSIC_CHRISTIAN_GOS_ALL', 'MUSIC_CLASSICAL_ALL', 'MUSIC_COUNTRY_ALL', 'MUSIC_JAZZ_ALL',
                          'MUSIC_ANY_ALL', 'MUSIC_RNB_ALL', 'MUSIC_ROCKNROLL_ALL', 'NUTRITION_NATURAL_FOODS_ALL',
                          'NUTRITION_NUTRITION_DIET_ALL', 'NUTRITION_VIT_SUPPLEMENTS_ALL',
                          'NUTRITION_WEIGHT_CONTROL_ALL', 'OTHER_ELECTRONICS_ALL', 'OTHER_GRANDHHILDREN_ALL',
                          'OTHER_ONLINE_HOUSEHOLD_ALL', 'OTHER_SCIENCE_TECHNOLOGY_ALL', 'OTHER_SWIMMING_POOL_ALL',
                          'PETS_OWN_CAT_ALL', 'PETS_OWN_DOG_ALL', 'PETS_PETS_ALL', 'SPORTS_BOATING_SAILING_ALL',
                          'SPORTS_CAMPING_HIKING_ALL', 'SPORTS_CYCLING_ALL', 'SPORTS_FISHING_ALL',
                          'SPORTS_FITNESS_EXERCISE_ALL', 'SPORTS_GOLF_ALL', 'SPORTS_HUNTING_BIG_GAME_ALL',
                          'SPORTS_HUNTING_SHOOTING_ALL', 'SPORTS_NASCAR_ALL', 'SPORTS_RUNNING_JOGGING_ALL',
                          'SPORTS_SKIING_SNOWBOARDING_ALL', 'SPORTS_SPORT_PARTICIPATION_ALL',
                          'SPORTS_WALKING_FOR_HEALTH_ALL', 'SPORTS_YOGA_PILATES_ALL', 'TRAVEL_BUSINESS_TRAVEL_ALL',
                          'TRAVEL_CASINO_GAMBLING_ALL', 'TRAVEL_CRUISESHIP_VACATION_ALL', 'TRAVEL_INTERNATIONAL_ALL',
                          'TRAVEL_LEISURE_TRAVEL_ALL', 'TRAVEL_RV_VACATIONS_ALL', 'TRAVEL_TIMESHARE_ALL',
                          'TRAVEL_TRAVEL_IN_THE_USA_ALL', 'TRAVEL_TRAVELER_ALL', 'OTHER_MILITARY_VETERAN_HH_ALL',
                          'OTHER_OWN_SMART_PHONE_ALL', 'FACEBOOK_USER_ALL', 'INSTAGRAM_USER_ALL', 'PINTEREST_USER_ALL',
                          'TWITTER_USER_ALL', 'MO_BUYER', 'HH_PURCHASE_CHANNEL_INT', 'HH_PURCHASE_CHANNEL_MO',
                          'HH_PURCHASE_CHANNEL_SWPSTAKE', 'HH_PURCHASE_CHANNEL_TELEMKT', 'CLUB_CONTINUITY_BUYER',
                          'PAYMENT_METHOD_CASH', 'PAYMENT_METHOD_CC', 'TYP_CC_AMERICAN_EXPRESS', 'TYP_CC_ANY_CC',
                          'TYP_CC_BANK_CARD', 'TYP_CC_CATALOG_SHOWROOM', 'TYP_CC_COMPUTER_ELECTRONIC',
                          'TYP_CC_DEBIT_CARD', 'TYP_CC_FIN_CO_CARD', 'TYP_CC_FURNITURE', 'TYP_CC_GROCERY',
                          'TYP_CC_HOME_IMPROVEMENT', 'TYP_CC_HOME_OFFICE_SUPPLY', 'TYP_CC_LOW_END_DEPART_STORE',
                          'TYP_CC_MAIN_STR_RETAIL', 'TYP_CC_MASTERCARD', 'TYP_CC_MEMBERSHIP_WAREHOUSE', 'TYP_CC_MISC',
                          'TYP_CC_OIL_GAS_CARD', 'TYP_CC_SPECIALTY_APPRL', 'TYP_CC_SPORTING_GOODS', 'TYP_CC_STD_RETAIL',
                          'TYP_CC_STD_SPECIALTY_CARD', 'TYP_CC_TRAVEL_ENTERTAINMENT', 'TYP_CC_TV_MO',
                          'TYP_CC_UPSCALE_RETAIL', 'TYP_CC_UPSCALE_SPEC_RETAIL', 'TYP_CC_VISA',
                          'LIFECYCLE_BABY_BOOMERS', 'LIFECYCLE_DINKS_DUAL_INCOME_NO_KIDS', 'LIFECYCLE_FAMILY_TIES',
                          'LIFECYCLE_GENERATION_X', 'LIFECYCLE_MILLENNIALS',
                          'LIFECYCLE_MILLENNIALS_BUTFIRSTLETMETAKEASELFIE', 'LIFECYCLE_MILLENNIALS_GETTINHITCHED',
                          'LIFECYCLE_MILLENNIALS_IMANADULT', 'LIFECYCLE_MILLENNIALS_LIVESWITHMOM',
                          'LIFECYCLE_MILLENNIALS_MOMLIFE', 'LIFECYCLE_MILLENNIALS_PUTTINGDOWNROOTS',
                          'VEHICLE_CATEGORY_LUXURY_CARS', 'VEHICLE_CATEGORY_PASSENGER_CARS',
                          'VEHICLE_CATEGORY_SPORT_CARS', 'VEHICLE_CATEGORY_SUV', 'VEHICLE_CATEGORY_TRUCKS',
                          'VEHICLE_CATEGORY_VANS', 'VEHICLE_SUB_CATEGORY_CARGO_VANS',
                          'VEHICLE_SUB_CATEGORY_COMPACT_PICKUPS', 'VEHICLE_SUB_CATEGORY_CONVERTIBLES',
                          'VEHICLE_SUB_CATEGORY_CROSSOVERS', 'VEHICLE_SUB_CATEGORY_EXOTICS',
                          'VEHICLE_SUB_CATEGORY_FULL_SIZE_PICKUPS', 'VEHICLE_SUB_CATEGORY_FULL_SIZE_SUVS',
                          'VEHICLE_SUB_CATEGORY_HEAVY_DUTY_PICKUPS', 'VEHICLE_SUB_CATEGORY_LARGE_CARS',
                          'VEHICLE_SUB_CATEGORY_LUXURY_SUVS', 'VEHICLE_SUB_CATEGORY_MIDSIZE_CARS',
                          'VEHICLE_SUB_CATEGORY_MIDSIZE_SUVS', 'VEHICLE_SUB_CATEGORY_MINI_SPORT_UTILITIES',
                          'VEHICLE_SUB_CATEGORY_MINIVANS', 'VEHICLE_SUB_CATEGORY_NEAR_LUXURY',
                          'VEHICLE_SUB_CATEGORY_OPEN_TOPS', 'VEHICLE_SUB_CATEGORY_PASSENGER_VANS',
                          'VEHICLE_SUB_CATEGORY_SMALL_CARS', 'VEHICLE_SUB_CATEGORY_SPECIALTY_TRUCKS',
                          'VEHICLE_SUB_CATEGORY_SPORTY_CARS', 'VEHICLE_SUB_CATEGORY_STATELY_WAGON',
                          'VEHICLE_SUB_CATEGORY_STATION_WAGONS', 'VEHICLE_SUB_CATEGORY_TRUE_LUXURY',
                          'MAIL_ORDER_RESPONDER_INSURANCE', 'COLLEGE_GRAD_TRIGGER', 'EMPTY_NESTER_TRIGGER',
                          'NEW_FIRST_CHILD_0_2_TRIGGER', 'HOME_MKT_VALUE_TRIGGER', 'INCOME_TRIGGER',
                          'NEW_ADULT_TO_FILE_TRIGGER', 'NEW_PRE_DRIVER_TRIGGER', 'NEW_YNG_ADULT_TO_FILE_TRIGGER',
                          'NEW_MARRIED_TRIGGER', 'NEW_SINGLE_TRIGGER', 'RETIRED_TRIGGER', 'VALUESCORE_TRIGGER'}

        df = self.epsilon_variable_recoder(df=df,
                                           required_columns=variable_set_1,
                                           mapping_dict={'Y': 1, 'N': 0})

        # meritscore recoding
        df = self.epsilon_variable_recoder(df=df,
                                           required_columns={'MERITSCORE'},
                                           mapping_dict={'A1': 12, 'A2': 11, 'B1': 10, 'B2': 9, 'C1': 8, 'C2': 7,
                                                         'D1': 6, 'D2': 5, 'D3': 4, 'E1': 3,
                                                         'E2': 2, 'E3': 1})

        # value_score recoding
        df = self.epsilon_variable_recoder(df=df,
                                           required_columns={'TARGET_VALUESCORE_20_ALL_MARKETERS',
                                                             'TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS',
                                                             'TARGET_VALUESCORE_20_BANK_CARD_MARKETERS',
                                                             'TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS',
                                                             'TRIGGERVAL_VALUESCORE'},
                                           mapping_dict={'A1': 12, 'A2': 11, 'B1': 10, 'B2': 9, 'C1': 8, 'C2': 7,
                                                         'D1': 6, 'D2': 5, 'D3': 4, 'E1': 3,
                                                         'E2': 2, 'E3': 1})
        # square_feet recoding
        df = self.epsilon_variable_recoder(df=df,
                                           required_columns={'LIVING_AREA_SQ_FTG_RANGE', 'TRIGGERVAL_HOME_MKT_VALUE'},
                                           mapping_dict={'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
                                                         'I': 9, 'J': 10, 'K': 11, 'L': 12,
                                                         'M': 13, 'N': 14,
                                                         'O': 15, 'P': 16, 'Q': 17, 'Z': None})
        # income recoding
        df = self.epsilon_variable_recoder(df=df,
                                           required_columns={'TRIGGERVAL_INCOME', 'ADV_TGT_INCOME_30',
                                                             'PROPERTY_LOT_SIZE_IN_ACRES'},
                                           mapping_dict={'A': 10, 'B': 11, 'C': 12, 'D': 13, '9': 9, '8': 8, '7': 7,
                                                         '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, '1': 1})
        df = self.epsilon_narrow_band_income_recoder(df)

        epsilon_recoded_variables = {'MERITSCORE', 'TARGET_VALUESCORE_20_ALL_MARKETERS',
                                     'TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS',
                                     'TARGET_VALUESCORE_20_BANK_CARD_MARKETERS',
                                     'TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS', 'TRIGGERVAL_VALUESCORE',
                                     'LIVING_AREA_SQ_FTG_RANGE',
                                     'TRIGGERVAL_HOME_MKT_VALUE', 'TRIGGERVAL_INCOME', 'ADV_TGT_INCOME_30',
                                     'PROPERTY_LOT_SIZE_IN_ACRES',
                                     'ADV_TGT_NARROW_BAND_INCOME_30'}

        epsilon_recoded_variables.update(variable_set_1)
        return df, {i.lower() for i in epsilon_recoded_variables}

    @staticmethod
    def epsilon_narrow_band_income_recoder(df: pd.DataFrame):
        df['ADV_TGT_NARROW_BAND_INCOME_30'] = df['ADV_TGT_NARROW_BAND_INCOME_30'].astype(str)

        def bandincome_rewritten_2(row):
            mapping_rules = {'nan': None, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15,
                             'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22,
                             'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32,
                             'X': 33, 'Y': 34, 'Z': 35}
            your_keyword = row
            if your_keyword in mapping_rules.keys():
                return mapping_rules[your_keyword]

        df['ADV_TGT_NARROW_BAND_INCOME_30_Epsilon_Recorded'] = df['ADV_TGT_NARROW_BAND_INCOME_30'].apply(
            lambda x: bandincome_rewritten_2(x))
        return df

    @staticmethod
    def epsilon_variable_recoder(df: pd.DataFrame,
                                 required_columns: set,
                                 mapping_dict: dict):
        if not isinstance(required_columns, set):
            required_columns = set(required_columns)

        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"{missing_cols} cannot be found in dataframe")
        else:
            for variable in required_columns:
                df[f"{variable}_Epsilon_Recoded"] = df[variable].map(mapping_dict)
        return df

    @staticmethod
    def find_eps_col(df):
        # converting to lower case
        df_col_nm = [i.lower() for i in list(df.columns.values)]
        eps_nm = ["file_code", "record_quality_code", "num_sourc_verify_hh", "fips_state_code", "zip", "zip4",
                  "delivery_point_code", "carrier_route", "contracted_address", "post_office_name", "state",
                  "county_code", "addr_quality_code", "addr_typ", "verification_date_hh", "surname", "agility_addr_key",
                  "agility_hh_key", "person_seq_no1", "ttl_code1", "given_name1", "middle_initital_1", "gender1",
                  "member_code_person1", "verification_date_person1", "african_american_conf_code1",
                  "assimilation_code1", "ethnic_group1", "ethnic_group_code1", "hisp_country_origin1", "language_code1",
                  "religion1", "num_tradelines1", "bankcard_issue_date1", "adv_ind_marital_status1",
                  "adv_ind_marital_stats_indicatr1", "bthday_person_w_day_enh1", "bthday_mth_indicator_enh1",
                  "individual_exact_age1", "self_reported_responder1", "agility_individual_key1", "person_seq_no2",
                  "ttl_code2", "given_name2", "middle_initital_2", "gender2", "member_code_person2",
                  "verification_date_person2", "african_american_conf_code2", "assimilation_code2", "ethnic_group2",
                  "ethnic_group_code2", "hisp_country_origin2", "language_code2", "religion2", "num_tradelines2",
                  "bankcard_issue_date2", "adv_ind_marital_status2", "adv_ind_marital_stats_indicatr2",
                  "bthday_person_w_day_enh2", "bthday_mth_indicator_enh2", "individual_exact_age2",
                  "self_reported_responder2", "agility_individual_key2", "person_seq_no3", "ttl_code3", "given_name3",
                  "middle_initital_3", "gender3", "member_code_person3", "verification_date_person3",
                  "african_american_conf_code3", "assimilation_code3", "ethnic_group3", "ethnic_group_code3",
                  "hisp_country_origin3", "language_code3", "religion3", "num_tradelines3", "bankcard_issue_date3",
                  "adv_ind_marital_status3", "adv_ind_marital_stats_indicatr3", "bthday_person_w_day_enh3",
                  "bthday_mth_indicator_enh3", "individual_exact_age3", "self_reported_responder3",
                  "agility_individual_key3", "person_seq_no4", "ttl_code4", "given_name4", "middle_initital_4",
                  "gender4", "member_code_person4", "verification_date_person4", "african_american_conf_code4",
                  "assimilation_code4", "ethnic_group4", "ethnic_group_code4", "hisp_country_origin4", "language_code4",
                  "religion4", "num_tradelines4", "bankcard_issue_date4", "adv_ind_marital_status4",
                  "adv_ind_marital_stats_indicatr4", "bthday_person_w_day_enh4", "bthday_mth_indicator_enh4",
                  "individual_exact_age4", "self_reported_responder4", "agility_individual_key4", "person_seq_no5",
                  "ttl_code5", "given_name5", "middle_initital_5", "gender5", "member_code_person5",
                  "verification_date_person5", "african_american_conf_code5", "assimilation_code5", "ethnic_group5",
                  "ethnic_group_code5", "hisp_country_origin5", "language_code5", "religion5", "num_tradelines5",
                  "bankcard_issue_date5", "adv_ind_marital_status5", "adv_ind_marital_stats_indicatr5",
                  "bthday_person_w_day_enh5", "bthday_mth_indicator_enh5", "individual_exact_age5",
                  "self_reported_responder5", "agility_individual_key5", "niches_40", "political_donor_props",
                  "political_party_hh", "occupation", "ocredit_auto_loans", "ocredit_edu_student_loans",
                  "ocredit_fin_services_banking", "ocredit_fin_services_install", "ocredit_fin_services_insurance",
                  "ocredit_leasing", "ocredit_home_mortg", "credit_active", "short_term_liability", "wealth_resources",
                  "investment_resources", "liquid_resources", "mortg_liability", "gold_wallets", "silver_wallets",
                  "platinum_wallets", "book_best_selling_fiction_all", "book_bible_devotional_all",
                  "book_books_reading_all", "books_sports_all", "book_children_book_all", "book_cooking_culinary_all",
                  "book_country_lifestyle_all", "book_entertainment_all", "book_fashion_all",
                  "book_interior_decorating_all", "book_medical_or_health_all", "book_military_all", "book_romance_all",
                  "book_world_news_all", "donor_donate_charit_causes_all", "donor_active_military_all",
                  "donor_alzheimer_all", "donor_animal_welfare_all", "donor_arts_cultural_all", "donor_cancer_all",
                  "donor_catholic_all", "donor_children_all", "donor_humanitarian_all", "donor_native_american_all",
                  "donor_other_religious_all", "donor_political_conserv_all", "donor_political_liberal_all",
                  "donor_veteran_all", "donor_world_relief_all", "donor_wildlife_envirn_caus_all", "collect_any_all",
                  "collect_art_antique_all", "collect_dolls_all", "collect_figurines_all", "collect_stamps_all",
                  "collect_coin_all", "hobbies_automotive_work_all", "hobbies_baking_all",
                  "hobbies_bird_feed_watch_all", "hobbies_career_advc_course_all", "hobbies_cigar_smoking_all",
                  "hobbies_contest_swpstakes_all", "hobbies_cooking_all", "hobbies_crafts_all",
                  "hobbies_cultural_events_all", "hobbies_gardening_all", "hobbies_gourmet_foods_all",
                  "hobbies_any_all", "hobbies_home_improv_diy_all", "hobbies_home_study_course_all",
                  "hobbies_motorcycle_riding_all", "hobbies_photography_all", "hobbies_quilting_all",
                  "hobbies_scrapbooking_all", "hobbies_self_improve_cozs_all", "hobbies_sewing_knitting_all",
                  "hobbies_wine_all", "hobbies_woodworking_all", "invest_insur_burial_insur_all",
                  "invest_insur_insurance_all", "invest_insur_investments_all", "invest_insur_juvlife_insur_all",
                  "invest_insur_life_insur_all", "invest_insur_medicare_cov_all", "invest_insur_mutual_funds_all",
                  "invest_insur_stocks_all", "mo_any_all", "mo_apprl_all", "mo_big_tall_all", "mo_books_all",
                  "mo_children_prod_all", "mo_food_all", "mo_gifts_all", "mo_health_beauty_prod_all",
                  "mo_home_furnishing_all", "mo_jewelry_all", "mo_magazines_all", "mo_videos_dvd_all",
                  "mo_women_plus_all", "music_christian_gos_all", "music_classical_all", "music_country_all",
                  "music_jazz_all", "music_any_all", "music_rnb_all", "music_rocknroll_all",
                  "nutrition_natural_foods_all", "nutrition_nutrition_diet_all", "nutrition_vit_supplements_all",
                  "nutrition_weight_control_all", "other_electronics_all", "other_grandhhildren_all",
                  "other_online_household_all", "other_science_technology_all", "other_swimming_pool_all",
                  "pets_own_cat_all", "pets_own_dog_all", "pets_pets_all", "sports_boating_sailing_all",
                  "sports_camping_hiking_all", "sports_cycling_all", "sports_fishing_all",
                  "sports_fitness_exercise_all", "sports_golf_all", "sports_hunting_big_game_all",
                  "sports_hunting_shooting_all", "sports_nascar_all", "sports_running_jogging_all",
                  "sports_skiing_snowboarding_all", "sports_sport_participation_all", "sports_walking_for_health_all",
                  "sports_yoga_pilates_all", "travel_business_travel_all", "travel_casino_gambling_all",
                  "travel_cruiseship_vacation_all", "travel_international_all", "travel_leisure_travel_all",
                  "travel_rv_vacations_all", "travel_timeshare_all", "travel_travel_in_the_usa_all",
                  "travel_traveler_all", "other_military_veteran_hh_all", "other_own_smart_phone_all",
                  "facebook_user_all", "instagram_user_all", "pinterest_user_all", "twitter_user_all",
                  "num_lifestyles_all", "num_lifestyles_self_rept", "adv_hh_edu_enh", "adv_hh_edu_indicator_enh",
                  "adv_hh_age_code_enh", "adv_hh_age_indicator_enh", "adv_hh_size_enh", "adv_hh_size_indicator_enh",
                  "adv_prez_children_enh", "adv_prez_children_indicatr_enh", "adv_num_adults",
                  "adv_num_adults_indicator", "adv_home_owner", "adv_home_owner_indicator", "adv_hh_marital_status",
                  "adv_hh_marital_status_indicatr", "adv_dwelling_typ", "adv_dwelling_typ_indicator",
                  "adv_length_residence", "adv_length_residence_indicator", "dwelling_typ_legacy", "age_18_24_spec_enh",
                  "age_25_34_spec_enh", "age_35_44_spec_enh", "age_45_54_spec_enh", "age_55_64_spec_enh",
                  "age_65_74_spec_enh", "prez_adult_75_above_enh", "prez_adult_unknown_age", "children_age_0_2_enh",
                  "children_age_3_5_enh", "children_age_6_10_enh", "children_age_11_15_enh", "children_age_16_17_enh",
                  "hh_typ_family_compos_enh", "num_children_hh_enh", "num_generations_hh_enh", "current_loan_to_value",
                  "adv_tgt_income_30", "adv_tgt_income_indicator_30", "adv_tgt_narrow_band_income_30", "meritscore",
                  "tgt_income_index_30", "tgt_net_worth_30", "tgt_pre_mover_20_model", "tgt_home_mkt_value_20",
                  "ava_home_equity_in_k", "first_mortg_amount_in_k", "home_equity_loan_date", "home_equity_loan_in_k",
                  "home_equity_loan_indicator", "home_mkt_value_tax_record", "home_sale_date", "home_sale_price_in_k",
                  "living_area_sq_ftg_range", "year_home_built", "mortg_date", "mortg_interest_rate_refin",
                  "mortg_interest_rate_typ_refin", "mortg_loan_typ_refin", "refin_indicator",
                  "second_mortg_amount_in_k", "original_mortg_amount", "property_lot_size_in_acres",
                  "exterior_wall_typ", "fuel_code", "home_heat_source", "roof_cover_typ", "structure_code",
                  "college_grad_change_date", "college_grad_trigger", "cc_change_date", "cc_trigger",
                  "triggerval_num_cc", "empty_nester_change_date", "empty_nester_trigger", "first_child_change_date",
                  "new_first_child_0_2_trigger", "home_mkt_value_change_date", "home_mkt_value_trigger",
                  "triggerval_home_mkt_value", "income_change_date", "income_trigger", "triggerval_income",
                  "new_adult_change_date", "new_adult_to_file_trigger", "new_driver_change_date",
                  "new_pre_driver_trigger", "new_yng_adult_to_change_date", "new_yng_adult_to_file_trigger",
                  "new_married_change_date", "new_married_trigger", "new_single_change_date", "new_single_trigger",
                  "retired_change_date", "retired_trigger", "valuescore_change_date", "valuescore_trigger",
                  "triggerval_valuescore", "niche_switch_trigger_date", "niche_switch_trigger",
                  "niche_switch_trigger_chg_typ", "act_avg_dollars_quint", "act_tot_dollars_quint",
                  "act_tot_offline_dollars_quint", "act_tot_online_dollars_quint", "act_num_purchase_quint",
                  "act_num_offline_purchase_quint", "act_num_online_purchase_quint", "eai_performance_rank", "mo_buyer",
                  "purchase_date_range", "channel_pref_rt_catalog_quint", "channel_pref_rt_online_quint",
                  "channel_pref_rt_retail_quint", "seasonality_fall_ratio_quint", "seasonality_spring_ratio_quint",
                  "seasonality_summer_ratio_quint", "seasonality_winter_ratio_quint", "hh_purchase_channel_int",
                  "hh_purchase_channel_mo", "hh_purchase_channel_swpstake", "hh_purchase_channel_telemkt",
                  "num_dollars_on_returns", "num_one_shot_orders", "num_returns", "num_returns_last_year",
                  "club_continuity_buyer", "payment_method_cash", "payment_method_cc", "arts_crafts_quint",
                  "arts_crafts_rcy_purch", "b2b_business_mkt_quint", "b2b_business_mkt_rcy_purch", "b2b_mro_quint",
                  "b2b_mro_rcy_purch", "b2b_office_co_gifts_quint", "b2b_office_co_gifts_rcy_purch",
                  "b2b_train_publications_quint", "b2b_train_pub_rcy_purch", "beauty_spa_quint", "beauty_spa_rcy_purch",
                  "beverages_quint", "beverages_rcy_purch", "books_quint", "books_rcy_purch", "children_quint",
                  "children_rcy_purch", "collect_quint", "collect_rcy_purch", "fashion_acc_beauty_quint",
                  "fashion_acc_beauty_rcy_purch", "f_m_apprl_quint", "f_m_apprl_rcy_purch", "garden_backyard_quint",
                  "garden_backyard_rcy_purch", "general_gifts_quint", "general_gifts_rcy_purch",
                  "health_vit_supplements_quint", "health_vit_sup_rcy_purch", "high_tkt_f_apprl_acc_quint",
                  "high_tkt_f_apprl_acc_rcy_purch", "high_tkt_home_decor_quint", "high_tkt_home_decor_rcy_purch",
                  "intimate_apprl_ug_quint", "intimate_apprl_ug_rcy_purch", "low_tkt_f_apprl_acc_quint",
                  "low_tkt_f_apprl_acc_rcy_purch", "low_to_mid_tkt_homedec_quint", "low_mid_tkt_homedec_rcy_purch",
                  "low_tkt_m_apprl_quint", "low_tkt_m_apprl_rcy_purch", "magazines_quint", "magazines_rcy_purch",
                  "mid_high_tkt_m_apprl_quint", "mid_high_tkt_m_apprl_rcy_purch", "mid_tkt_f_apprl_acc_quint",
                  "mid_tkt_f_apprl_acc_rcy_purch", "modern_decor_gifts_quint", "modern_decor_gift_rcy_purch",
                  "music_videos_quint", "music_videos_rcy_purch", "newsletters_quint", "newsletters_rcy_purch",
                  "publish_kitchen_home_quint", "publish_kitchen_home_rcy_purch", "sr_prod_quint", "sr_prod_rcy_purch",
                  "shoes_quint", "shoes_rcy_purch", "spec_food_gift_quint", "spec_food_gift_rcy_purch",
                  "sports_outdoor_quint", "sports_outdoor_rcy_purch", "sports_merch_actwear_quint",
                  "sports_merch_actwear_rcy_purch", "tools_elec_quint", "tools_elec_rcy_purch", "m_apprl_quint",
                  "m_apprl_rcy_purch", "oth_quint", "oth_rcy_purch", "mt_online_transactor", "mt_vehicle_purchr",
                  "mt_auto_loan_purchr", "mt_401k_owners", "mt_debit_card_user", "mt_online_broker_user",
                  "mt_online_savings_user", "mt_low_interest_cc_user", "mt_rewards_card_cash_back_user",
                  "mt_rewards_card_oth_user", "mt_national_bank_custr", "mt_regional_bank_custr",
                  "mt_community_bank_custr", "mt_credit_union_memb", "mt_interest_checkingpref_custr",
                  "mt_fin_advisor_custr", "mt_cc_revolvers", "mt_non_401k_mf_investors", "mt_non_401k_stk_bd_investors",
                  "mt_debit_card_user2", "mt_invest_trustbankpref_custr", "mt_likely_have_mortg",
                  "mt_loyal_fin_institution_custr", "mt_online_insurance_buyer", "mt_second_homeowners",
                  "mt_timeshare_owners", "mt_fin_institution_shoppers", "mt_weekly_online_bankers", "mt_freq_atm_custr",
                  "mt_deposit_custr", "mt_lending_custr", "mt_convenience_custr", "mt_cc_attrition_hh",
                  "mt_edu_savings_plan_owners", "mt_student_loan_custr", "mt_active_on_fb",
                  "mt_active_on_fb_brand_likers", "mt_active_on_fb_cate_recomm", "mt_tablet_owners",
                  "mt_social_media_pref_custr", "mt_active_on_pinterest", "mt_active_on_twitter",
                  "mt_in_mkt_to_purchase_a_home", "mt_in_mkt_to_get_a_home_loan", "mt_prim_cell_phone_users",
                  "mt_wired_service_custr", "mt_satellite_bundle_satellite_i_net_home_or_wireless",
                  "propensity_to_buy_lux_veh_suv", "propensity_buy_midsize_car", "propensity_buy_midsize_suv",
                  "likely_to_lease_vehicle", "typ_cc_american_express", "typ_cc_any_cc", "typ_cc_bank_card",
                  "typ_cc_catalog_showroom", "typ_cc_computer_electronic", "typ_cc_debit_card", "typ_cc_fin_co_card",
                  "typ_cc_furniture", "typ_cc_grocery", "typ_cc_home_improvement", "typ_cc_home_office_supply",
                  "typ_cc_low_end_depart_store", "typ_cc_main_str_retail", "typ_cc_mastercard",
                  "typ_cc_membership_warehouse", "typ_cc_misc", "typ_cc_oil_gas_card", "typ_cc_specialty_apprl",
                  "typ_cc_sporting_goods", "typ_cc_std_retail", "typ_cc_std_specialty_card",
                  "typ_cc_travel_entertainment", "typ_cc_tv_mo", "typ_cc_upscale_retail", "typ_cc_upscale_spec_retail",
                  "typ_cc_visa", "totalist_matchcode", "dma", "house_number", "fraction", "street_prefix_direction",
                  "street_name", "street_suffix", "street_post_direction", "route_designator_and_number",
                  "box_designator_and_number", "secondary_unit_designation", "political_party_individual_1",
                  "political_party_individual_2", "political_party_individual_3", "political_party_individual_4",
                  "political_party_individual_5", "birthday_day_indicator_no_restricted_source_enhanced_1",
                  "birthday_day_indicator_no_restricted_source_enhanced_2",
                  "birthday_day_indicator_no_restricted_source_enhanced_3",
                  "birthday_day_indicator_no_restricted_source_enhanced_4",
                  "birthday_day_indicator_no_restricted_source_enhanced_5", "person_key_1", "person_key_2",
                  "person_key_3", "person_key_4", "person_key_5", "agility_occupancy_score_1",
                  "agility_occupancy_score_2", "agility_occupancy_score_3", "agility_occupancy_score_4",
                  "agility_occupancy_score_5", "ethnic_group_code_household", "ethnic_household",
                  "birthdate_of_1st_child_enhanced", "birthdate_of_1st_child_indicator_enhanced",
                  "gender_of_child_1_enhanced", "birthdate_of_2nd_child_enhanced",
                  "birthdate_of_2nd_child_indicator_enhanced", "gender_of_child_2_enhanced",
                  "birthdate_of_3rd_child_enhanced", "birthdate_of_3rd_child_indicator_enhanced",
                  "gender_of_child_3_enhanced", "birthdate_of_4th_child_enhanced",
                  "birthdate_of_4th_child_indicator_enhanced", "gender_of_child_4_enhanced", "lifecycle_baby_boomers",
                  "lifecycle_dinks_dual_income_no_kids", "lifecycle_family_ties", "lifecycle_generation_x",
                  "lifecycle_millennials", "lifecycle_millennials_butfirstletmetakeaselfie",
                  "lifecycle_millennials_gettinhitched", "lifecycle_millennials_imanadult",
                  "lifecycle_millennials_liveswithmom", "lifecycle_millennials_momlife",
                  "lifecycle_millennials_puttingdownroots", "target_valuescore_20_all_marketers",
                  "target_valuescore_20_auto_finance_marketers", "target_valuescore_20_bank_card_marketers",
                  "target_valuescore_20_retail_card_marketers", "rooftop_latitudelongitude_indicator",
                  "rooftop_latitude", "rooftop_longitude", "mt_aca_health_insurance_purchasers", "mt_airline_upgraders",
                  "mt_amazon_prime_customers", "mt_animal_welfare_donors", "mt_annuity_customers",
                  "mt_att_cell_phone_customer", "mt_auto_insurance_agent_sold", "mt_auto_insurance_call_center_sold",
                  "mt_auto_insurance_self_serve_online_buyers",
                  "mt_auto_insurance_prem_loyalty_gifts_telematics_customers",
                  "mt_auto_insurance_premium_discount_via_telematics_customer", "mt_auto_warranty_purchasers",
                  "mt_avid_book_readers", "mt_avid_magazine_readers", "mt_bar_and_lounge_food_enthusiasts",
                  "mt_bargain_hotel_shoppers", "mt_bargain_shoppers", "mt_baseball_enthusiasts",
                  "mt_basketball_enthusiasts", "mt_bible_devotional_readers", "mt_boat_owners", "mt_book_readers",
                  "mt_brand_driven_home_cleaners", "mt_brand_loyalists", "mt_brand_motivated_laundry_users",
                  "mt_brand_motivated_personal_care_product_users", "mt_branded_retail_credit_card_users",
                  "mt_budget_meal_planners", "mt_burial_insurance_purchaser", "mt_business_traveler",
                  "mt_cable_bundle_cable_internet_home_phone", "mt_cable_tv_premium_subscribers", "mt_cancer_donors",
                  "mt_carry_out_enthusiasts", "mt_casino_gamer", "mt_casual_dining_enthusiasts",
                  "mt_catering_delivery_customers", "mt_catering_pick_up_customers",
                  "mt_certificates_of_deposit_customers", "mt_childrens_causes_donors",
                  "mt_christian_or_gospel_music_enthusiasts", "mt_christmas_ornamentscollectibles_buyer",
                  "mt_cigarpipe_enthusiasts", "mt_click_to_cart_home_delivery_customers",
                  "mt_click_to_cart_pick_up_customers", "mt_coffee_enthusiasts", "mt_coins_collector",
                  "mt_conservative_causes_donors", "mt_conservative_investment_style_consumers", "mt_convenience_cook",
                  "mt_convenience_driven_personal_care_product_users", "mt_convenience_home_cleaners",
                  "mt_cord_cutters", "mt_country_music_enthusiasts", "mt_credit_card_balance_transfer_users",
                  "mt_cultural_arts__events_attendees", "mt_debit_card_rewards_users", "mt_democratic_voter",
                  "mt_diet_conscious_households", "mt_direct_media_preference_customers", "mt_do_it_yourselfer",
                  "mt_employer_provided_health_insurance_policy_holders", "mt_entertainment_readers",
                  "mt_environmental_donors", "mt_environmentally_focused_household", "mt_everyday_low_price_shoppers",
                  "mt_experimental_cooks", "mt_extreme_fitness_enthusiasts",
                  "mt_financialhealth_newsletter_subscribers", "mt_fine_dining_enthusiasts", "mt_football_enthusiasts",
                  "mt_frequent_mobile_purchasers", "mt_frequent_online_movie_viewers",
                  "mt_frequent_online_music_purchasers", "mt_frequent_takeout_food_hh", "mt_fresh_food_seekers",
                  "mt_future_investors", "mt_gamers", "mt_green_product_purchasers",
                  "mt_grocery_loyalty_card_customers", "mt_heavy_cleaner", "mt_heavy_coupon_users",
                  "mt_heavy_fiber_focused_food_buyers", "mt_heavy_gluten_free_food_buyers",
                  "mt_heavy_low_fat_food_buyers", "mt_high_dollar_other_causes_non_religious_donor",
                  "mt_high_dollar_religious_causes_donor", "mt_high_end_shoppers",
                  "mt_home_cleaning_new_product_seekers", "mt_home_warranty_purchasers",
                  "mt_hotel_loyalty_program_members", "mt_identity_theft_protection_purchasers",
                  "mt_impulse_purchasers", "mt_incentive_seekers", "mt_independent_voters", "mt_insurance_switcher",
                  "mt_international_traveler", "mt_international_wireless_or_landline_customers",
                  "mt_internet_research_preference_customers", "mt_job_switchers", "mt_kroger_enthusiasts",
                  "mt_label_readers", "mt_latin_music_enthusiasts", "mt_laundry_new_product_seekers",
                  "mt_liberal_causes_donors", "mt_likely_cruiser", "mt_likely_mortgage_refinancers",
                  "mt_likely_planned_givers", "mt_likely_to_suffer_from_insomnia",
                  "mt_likely_to_use_an_investment_broker", "mt_likely_voters", "mt_live_music_concert_attendees",
                  "mt_long_term_care", "mt_magazine_readers", "mt_master_cook", "mt_meal_planners",
                  "mt_medicaid_potential_qualified_household", "mt_medicare_advantage_plan_purchasers",
                  "mt_medicare_dual_eligible_household", "mt_medicare_plan_d_prescription_drug_health_purchaser",
                  "mt_medicare_supplement_insurance_purchasers", "mt_mens_big_and_tall_apparel_customers",
                  "mt_midmarket_term_life_insurance_purchasers", "mt_midmarket_whole_life_insurance_purchasers",
                  "mt_mobile_banking_users", "mt_mobile_browsers", "mt_mobile_shopping_list_users",
                  "mt_monitored_home_security_system_owners", "mt_multi_policy_insurance_owners",
                  "mt_multi_retailer_shoppers", "mt_nascar_enthusiasts", "mt_natural__green_product_home_cleaners",
                  "mt_natural_product_personal_care_product_users", "mt_new_luxury_vehicle_purchasers",
                  "mt_new_non_luxury_vehicle_purchasers", "mt_new_roof_customers", "mt_on_demand_movie_subscribers",
                  "mt_one_stop_shoppers", "mt_online_delivery_restaurant_customers",
                  "mt_online_pick_up_restaurant_customers", "mt_online_degreeeducation_seekers",
                  "mt_online_home_cleaning_product_buyers", "mt_online_laundry_product_buyers",
                  "mt_online_magazinenewspaper_subscribers", "mt_online_personal_care_product_buyers",
                  "mt_online_pet_food_buyers", "mt_organic_food_purchasers", "mt_organic_product_purchasers",
                  "mt_paper_shopping_list_users", "mt_paycheck_to_paycheck_consumers",
                  "mt_personal_care_new_product_seekers", "mt_personal_traveler", "mt_pet_owners", "mt_pets_are_family",
                  "mt_plan_to_get_fitness_membership", "mt_plan_to_purchase_home_security_systems",
                  "mt_pre_shop_planners", "mt_premium_natural_home_cleaners",
                  "mt_premium_natural_laundry_product_buyers", "mt_premium_natural_personal_care_product_users",
                  "mt_prepaid_card_owners", "mt_price_driven_home_cleaners", "mt_price_matchers",
                  "mt_price_motivated_laundry_product_users", "mt_price_motivated_personal_care_product_users",
                  "mt_private_label_shopper", "mt_professional_sports_events_attendees",
                  "mt_public_transportation_users", "mt_quantum_upgrade_customers",
                  "mt_quick_service_restaurant_enthusiasts", "mt_quick_shop_at_walmart_or_target",
                  "mt_real_ingredient_cook", "mt_renters_and_auto_insurance_joint_policy_holders",
                  "mt_republican_voter", "mt_restaurant_app_users", "mt_restaurant_loyalty_app_users",
                  "mt_restaurant_loyalty_card_customers", "mt_retail_texters", "mt_retailer_circular_readers",
                  "mt_retailer_email_subscribers", "mt_retired_but_still_working", "mt_romance_readers",
                  "mt_satellite_radio_subscribers", "mt_scent_seekers", "mt_self_pay_health_insurance",
                  "mt_self_insured_dental_customers", "mt_senior_caregivers", "mt_senior_living_searchers",
                  "mt_smart_phone_user", "mt_soccer_enthusiasts", "mt_solar_roofing_interest", "mt_sports_readers",
                  "mt_sprint_cell_phone_customer", "mt_stock_up_at_grocery_stores", "mt_stock_up_at_walmart",
                  "mt_stock_up_shoppers", "mt_subscription_or_auto_shipment_customers", "mt_swing_voters",
                  "mt_t_mobile_cell_phone_customer", "mt_target_cartwheel_users", "mt_target_enthusiast",
                  "mt_technology_early_adopters", "mt_term_life", "mt_uberlyft_users", "mt_ubi_purchaser",
                  "mt_underbanked_consumers", "mt_uninsured_for_health", "mt_unscented_product_seekers",
                  "mt_upcoming_retirees_50_64", "mt_upcoming_retirees_65_and_older", "mt_vacation_spenders",
                  "mt_value_chains_enthusiasts", "mt_vehicle_diyrs", "mt_vehicle_service_center_users",
                  "mt_verizon_cell_phone_customer", "mt_veteran_donors", "mt_veterinarian_influenced_pet_owners",
                  "mt_voip_landlinecustomers", "mt_walmart_enthusiast", "mt_walmart_saving_catcher_users",
                  "mt_wearable_technology_users", "mt_web_and_brick__mortar_vieweronline_purchasers",
                  "mt_web_surferbrick__mortar_purchasers", "mt_wellness_households_health", "mt_whats_on_sale_shoppers",
                  "mt_whole_life", "mt_wired_line_video_connectors", "mt_womens_plus_size_apparel_customers",
                  "mt_work_for_small_company_offering_health_insurance", "mt_yogapilates_enthusiast",
                  "mt_breakfast_dining_enthusiasts", "mt_lunch_dining_enthusiasts", "mt_qsr_cash_customers",
                  "mt_meal_combo_consumers", "mt_low_sodium_consumers", "mt_dinner_dining_enthusiasts",
                  "mt_fresh_food_delivery_consumers", "mt_grocery_store_frequenters", "mt_grocery_store_app_users",
                  "mt_vegetarians", "mt_frequent_movie_enthusiasts", "mt_movie_loyalty_program_members",
                  "mt_opening_weekend_movie_enthusiasts", "mt_paid_streaming_enthusiasts",
                  "mt_free_streaming_enthusiasts", "mt_smart_tv_owners", "mt_home_shopping_network_enthusiasts",
                  "mt_mobile_phone_service_switchers", "mt_discount_movie_enthusiasts", "mt_apple_smart_phone_owners",
                  "mt_android_smart_phone_owners", "mt_art_house_movie_enthusiasts", "mt_fantasy_sports_enthusiasts",
                  "mt_home_remodelers", "mt_meditation_enthusiast", "mt_meal_kit_delivery_consumers",
                  "vehicle_category_luxury_cars", "vehicle_category_passenger_cars", "vehicle_category_sport_cars",
                  "vehicle_category_suv", "vehicle_category_trucks", "vehicle_category_vans",
                  "vehicle_insurance_renewal_month", "vehicle_sub_category_cargo_vans",
                  "vehicle_sub_category_compact_pickups", "vehicle_sub_category_convertibles",
                  "vehicle_sub_category_crossovers", "vehicle_sub_category_exotics",
                  "vehicle_sub_category_full_size_pickups", "vehicle_sub_category_full_size_suvs",
                  "vehicle_sub_category_heavy_duty_pickups", "vehicle_sub_category_large_cars",
                  "vehicle_sub_category_luxury_suvs", "vehicle_sub_category_midsize_cars",
                  "vehicle_sub_category_midsize_suvs", "vehicle_sub_category_mini_sport_utilities",
                  "vehicle_sub_category_minivans", "vehicle_sub_category_near_luxury", "vehicle_sub_category_open_tops",
                  "vehicle_sub_category_passenger_vans", "vehicle_sub_category_small_cars",
                  "vehicle_sub_category_specialty_trucks", "vehicle_sub_category_sporty_cars",
                  "vehicle_sub_category_stately_wagon", "vehicle_sub_category_station_wagons",
                  "vehicle_sub_category_true_luxury", "vehicle_make", "vehicle_model", "vehicle_year",
                  "mail_order_responder_insurance", "likely_to_buy_domestic_vehicle", "likely_to_buy_import_vehicle",
                  "likely_to_buy_new_vehicle", "likely_to_buy_used_vehicle", "likely_to_purchase_same_manufacturer",
                  "likely_to_use_dealer_service", "propensity_to_buy_compact_truck", "propensity_to_buy_economy_car",
                  "propensity_to_buy_economy_suv", "propensity_to_buy_full_size_truck",
                  "propensity_to_buy_luxury_truck_full_size", "business_flag", "company_name",
                  "number_of_cars_in_household", "number_of_trucks_in_household", "number_of_vehicles_in_household",
                  "vehicle_class_code_1", "vehicle_class_code_2", "vehicle_class_code_3", "vehicle_class_code_4",
                  "vehicle_class_code_5", "vehicle_class_code_6", "vehicle_first_seen_date_1",
                  "vehicle_first_seen_date_2", "vehicle_first_seen_date_3", "vehicle_first_seen_date_4",
                  "vehicle_first_seen_date_5", "vehicle_first_seen_date_6", "vehicle_fuel_type_code_1",
                  "vehicle_fuel_type_code_2", "vehicle_fuel_type_code_3", "vehicle_fuel_type_code_4",
                  "vehicle_fuel_type_code_5", "vehicle_fuel_type_code_6", "vehicle_last_seen_date_1",
                  "vehicle_last_seen_date_2", "vehicle_last_seen_date_3", "vehicle_last_seen_date_4",
                  "vehicle_last_seen_date_5", "vehicle_last_seen_date_6", "vehicle_make_1", "vehicle_make_2",
                  "vehicle_make_3", "vehicle_make_4", "vehicle_make_5", "vehicle_make_6",
                  "vehicle_manufacturing_code_1", "vehicle_manufacturing_code_2", "vehicle_manufacturing_code_3",
                  "vehicle_manufacturing_code_4", "vehicle_manufacturing_code_5", "vehicle_manufacturing_code_6",
                  "vehicle_mileage_code_1", "vehicle_mileage_code_2", "vehicle_mileage_code_3",
                  "vehicle_mileage_code_4", "vehicle_mileage_code_5", "vehicle_mileage_code_6", "vehicle_model_1",
                  "vehicle_model_2", "vehicle_model_3", "vehicle_model_4", "vehicle_model_5", "vehicle_model_6",
                  "vehicle_service_indicator_1", "vehicle_service_indicator_2", "vehicle_service_indicator_3",
                  "vehicle_service_indicator_4", "vehicle_service_indicator_5", "vehicle_service_indicator_6",
                  "vehicle_style_code_1", "vehicle_style_code_2", "vehicle_style_code_3", "vehicle_style_code_4",
                  "vehicle_style_code_5", "vehicle_style_code_6", "vehicle_year_1", "vehicle_year_2", "vehicle_year_3",
                  "vehicle_year_4", "vehicle_year_5", "vehicle_year_6", "ailments_addadhd_self_reported",
                  "ailments_allergies_self_reported", "ailments_anxiety_self_reported",
                  "ailments_arthritis_rheumatoid_self_reported", "ailments_arthritis_self_reported",
                  "ailments_back_pain_self_reported", "ailments_bladderbowel_self_reported",
                  "ailments_copd_self_reported", "ailments_depression_self_reported", "ailments_diabetes_self_reported",
                  "ailments_diabetes_type_1_self_reported", "ailments_diabetes_type_2_self_reported",
                  "ailments_digestive_self_reported", "ailments_foot_ailments_self_reported",
                  "ailments_hearing_loss_self_reported", "ailments_heart_condition_self_reported",
                  "ailments_high_blood_pressure_self_reported", "ailments_high_cholesterol_self_reported",
                  "ailments_insomnia_self_reported", "ailments_menopause_self_reported",
                  "ailments_osteoporosis_self_reported", "ailments_pain_self_reported",
                  "ailments_respiratory_ailments_self_reported", "ailments_sinusnasal_self_reported",
                  "ailments_vision_care_and_conditions_self_reported", "_2010_complete_census_geo_", "file_id",
                  "install_date"]
        if len(set(df_col_nm) - set(eps_nm)) <= 5:
            warnings.warn('espilon columns found')
            return True
        else:
            return False

    @staticmethod
    def epsilonrecoding(data):

        data.columns = map(str.upper, data.columns)

        # Record 1 and Null (1 & Null)
        variable_list1 = ['OCREDIT_AUTO_LOANS', 'OCREDIT_EDU_STUDENT_LOANS', 'OCREDIT_FIN_SERVICES_BANKING',
                          'OCREDIT_FIN_SERVICES_INSTALL',
                          'OCREDIT_FIN_SERVICES_INSURANCE', 'OCREDIT_LEASING', 'OCREDIT_HOME_MORTG', 'CREDIT_ACTIVE',
                          'GOLD_WALLETS', 'SILVER_WALLETS', 'PLATINUM_WALLETS', 'BOOK_BEST_SELLING_FICTION_ALL',
                          'BOOK_BIBLE_DEVOTIONAL_ALL', 'BOOK_BOOKS_READING_ALL', 'BOOKS_SPORTS_ALL',
                          'BOOK_CHILDREN_BOOK_ALL', 'BOOK_COOKING_CULINARY_ALL', 'BOOK_COUNTRY_LIFESTYLE_ALL',
                          'BOOK_ENTERTAINMENT_ALL', 'BOOK_FASHION_ALL', 'BOOK_INTERIOR_DECORATING_ALL',
                          'BOOK_MEDICAL_OR_HEALTH_ALL', 'BOOK_MILITARY_ALL', 'BOOK_ROMANCE_ALL', 'BOOK_WORLD_NEWS_ALL',
                          'DONOR_DONATE_CHARIT_CAUSES_ALL', 'DONOR_ACTIVE_MILITARY_ALL', 'DONOR_ALZHEIMER_ALL',
                          'DONOR_ANIMAL_WELFARE_ALL', 'DONOR_ARTS_CULTURAL_ALL',
                          'DONOR_CANCER_ALL', 'DONOR_CATHOLIC_ALL', 'DONOR_CHILDREN_ALL', 'DONOR_HUMANITARIAN_ALL',
                          'DONOR_NATIVE_AMERICAN_ALL', 'DONOR_OTHER_RELIGIOUS_ALL', 'DONOR_POLITICAL_CONSERV_ALL',
                          'DONOR_POLITICAL_LIBERAL_ALL', 'DONOR_VETERAN_ALL', 'DONOR_WORLD_RELIEF_ALL',
                          'DONOR_WILDLIFE_ENVIRN_CAUS_ALL', 'COLLECT_ANY_ALL', 'COLLECT_ART_ANTIQUE_ALL',
                          'COLLECT_DOLLS_ALL', 'COLLECT_FIGURINES_ALL', 'COLLECT_STAMPS_ALL', 'COLLECT_COIN_ALL',
                          'HOBBIES_AUTOMOTIVE_WORK_ALL', 'HOBBIES_BAKING_ALL', 'HOBBIES_BIRD_FEED_WATCH_ALL',
                          'HOBBIES_CAREER_ADVC_COURSE_ALL', 'HOBBIES_CIGAR_SMOKING_ALL',
                          'HOBBIES_CONTEST_SWPSTAKES_ALL', 'HOBBIES_COOKING_ALL', 'HOBBIES_CRAFTS_ALL',
                          'HOBBIES_CULTURAL_EVENTS_ALL', 'HOBBIES_GARDENING_ALL', 'HOBBIES_GOURMET_FOODS_ALL',
                          'HOBBIES_ANY_ALL', 'HOBBIES_HOME_IMPROV_DIY_ALL', 'HOBBIES_HOME_STUDY_COURSE_ALL',
                          'HOBBIES_MOTORCYCLE_RIDING_ALL', 'HOBBIES_PHOTOGRAPHY_ALL', 'HOBBIES_QUILTING_ALL',
                          'HOBBIES_SCRAPBOOKING_ALL', 'HOBBIES_SELF_IMPROVE_COZS_ALL',
                          'HOBBIES_SEWING_KNITTING_ALL', 'HOBBIES_WINE_ALL', 'HOBBIES_WOODWORKING_ALL',
                          'INVEST_INSUR_BURIAL_INSUR_ALL', 'INVEST_INSUR_INSURANCE_ALL',
                          'INVEST_INSUR_INVESTMENTS_ALL', 'INVEST_INSUR_JUVLIFE_INSUR_ALL',
                          'INVEST_INSUR_LIFE_INSUR_ALL', 'INVEST_INSUR_MEDICARE_COV_ALL',
                          'INVEST_INSUR_MUTUAL_FUNDS_ALL',
                          'INVEST_INSUR_STOCKS_ALL', 'MO_ANY_ALL', 'MO_APPRL_ALL', 'MO_BIG_TALL_ALL', 'MO_BOOKS_ALL',
                          'MO_CHILDREN_PROD_ALL', 'MO_FOOD_ALL', 'MO_GIFTS_ALL', 'MO_HEALTH_BEAUTY_PROD_ALL',
                          'MO_HOME_FURNISHING_ALL', 'MO_JEWELRY_ALL',
                          'MO_MAGAZINES_ALL', 'MO_VIDEOS_DVD_ALL', 'MO_WOMEN_PLUS_ALL', 'MUSIC_CHRISTIAN_GOS_ALL',
                          'MUSIC_CLASSICAL_ALL', 'MUSIC_COUNTRY_ALL', 'MUSIC_JAZZ_ALL', 'MUSIC_ANY_ALL',
                          'MUSIC_RNB_ALL',
                          'MUSIC_ROCKNROLL_ALL', 'NUTRITION_NATURAL_FOODS_ALL', 'NUTRITION_NUTRITION_DIET_ALL',
                          'NUTRITION_VIT_SUPPLEMENTS_ALL', 'NUTRITION_WEIGHT_CONTROL_ALL', 'OTHER_ELECTRONICS_ALL',
                          'OTHER_GRANDHHILDREN_ALL',
                          'OTHER_ONLINE_HOUSEHOLD_ALL', 'OTHER_SCIENCE_TECHNOLOGY_ALL', 'OTHER_SWIMMING_POOL_ALL',
                          'PETS_OWN_CAT_ALL', 'PETS_OWN_DOG_ALL', 'PETS_PETS_ALL',
                          'SPORTS_BOATING_SAILING_ALL', 'SPORTS_CAMPING_HIKING_ALL', 'SPORTS_CYCLING_ALL',
                          'SPORTS_FISHING_ALL', 'SPORTS_FITNESS_EXERCISE_ALL', 'SPORTS_GOLF_ALL',
                          'SPORTS_HUNTING_BIG_GAME_ALL',
                          'SPORTS_HUNTING_SHOOTING_ALL', 'SPORTS_NASCAR_ALL', 'SPORTS_RUNNING_JOGGING_ALL',
                          'SPORTS_SKIING_SNOWBOARDING_ALL', 'SPORTS_SPORT_PARTICIPATION_ALL',
                          'SPORTS_WALKING_FOR_HEALTH_ALL', 'SPORTS_YOGA_PILATES_ALL', 'TRAVEL_BUSINESS_TRAVEL_ALL',
                          'TRAVEL_CASINO_GAMBLING_ALL', 'TRAVEL_CRUISESHIP_VACATION_ALL', 'TRAVEL_INTERNATIONAL_ALL',
                          'TRAVEL_LEISURE_TRAVEL_ALL', 'TRAVEL_RV_VACATIONS_ALL', 'TRAVEL_TIMESHARE_ALL',
                          'TRAVEL_TRAVEL_IN_THE_USA_ALL', 'TRAVEL_TRAVELER_ALL', 'OTHER_MILITARY_VETERAN_HH_ALL',
                          'OTHER_OWN_SMART_PHONE_ALL',
                          'FACEBOOK_USER_ALL', 'INSTAGRAM_USER_ALL', 'PINTEREST_USER_ALL', 'TWITTER_USER_ALL',
                          'MO_BUYER', 'HH_PURCHASE_CHANNEL_INT', 'HH_PURCHASE_CHANNEL_MO',
                          'HH_PURCHASE_CHANNEL_SWPSTAKE', 'HH_PURCHASE_CHANNEL_TELEMKT', 'CLUB_CONTINUITY_BUYER',
                          'PAYMENT_METHOD_CASH', 'PAYMENT_METHOD_CC', 'TYP_CC_AMERICAN_EXPRESS', 'TYP_CC_ANY_CC',
                          'TYP_CC_BANK_CARD', 'TYP_CC_CATALOG_SHOWROOM', 'TYP_CC_COMPUTER_ELECTRONIC',
                          'TYP_CC_DEBIT_CARD',
                          'TYP_CC_FIN_CO_CARD', 'TYP_CC_FURNITURE', 'TYP_CC_GROCERY', 'TYP_CC_HOME_IMPROVEMENT',
                          'TYP_CC_HOME_OFFICE_SUPPLY', 'TYP_CC_LOW_END_DEPART_STORE', 'TYP_CC_MAIN_STR_RETAIL',
                          'TYP_CC_MASTERCARD', 'TYP_CC_MEMBERSHIP_WAREHOUSE', 'TYP_CC_MISC', 'TYP_CC_OIL_GAS_CARD',
                          'TYP_CC_SPECIALTY_APPRL', 'TYP_CC_SPORTING_GOODS',
                          'TYP_CC_STD_RETAIL', 'TYP_CC_STD_SPECIALTY_CARD', 'TYP_CC_TRAVEL_ENTERTAINMENT',
                          'TYP_CC_TV_MO', 'TYP_CC_UPSCALE_RETAIL', 'TYP_CC_UPSCALE_SPEC_RETAIL',
                          'TYP_CC_VISA', 'LIFECYCLE_BABY_BOOMERS', 'LIFECYCLE_DINKS_DUAL_INCOME_NO_KIDS',
                          'LIFECYCLE_FAMILY_TIES', 'LIFECYCLE_GENERATION_X',
                          'LIFECYCLE_MILLENNIALS', 'LIFECYCLE_MILLENNIALS_BUTFIRSTLETMETAKEASELFIE',
                          'LIFECYCLE_MILLENNIALS_GETTINHITCHED', 'LIFECYCLE_MILLENNIALS_IMANADULT',
                          'LIFECYCLE_MILLENNIALS_LIVESWITHMOM', 'LIFECYCLE_MILLENNIALS_MOMLIFE',
                          'LIFECYCLE_MILLENNIALS_PUTTINGDOWNROOTS', 'VEHICLE_CATEGORY_LUXURY_CARS',
                          'VEHICLE_CATEGORY_PASSENGER_CARS', 'VEHICLE_CATEGORY_SPORT_CARS', 'VEHICLE_CATEGORY_SUV',
                          'VEHICLE_CATEGORY_TRUCKS', 'VEHICLE_CATEGORY_VANS',
                          'VEHICLE_SUB_CATEGORY_CARGO_VANS', 'VEHICLE_SUB_CATEGORY_COMPACT_PICKUPS',
                          'VEHICLE_SUB_CATEGORY_CONVERTIBLES', 'VEHICLE_SUB_CATEGORY_CROSSOVERS',
                          'VEHICLE_SUB_CATEGORY_EXOTICS', 'VEHICLE_SUB_CATEGORY_FULL_SIZE_PICKUPS',
                          'VEHICLE_SUB_CATEGORY_FULL_SIZE_SUVS',
                          'VEHICLE_SUB_CATEGORY_HEAVY_DUTY_PICKUPS', 'VEHICLE_SUB_CATEGORY_LARGE_CARS',
                          'VEHICLE_SUB_CATEGORY_LUXURY_SUVS', 'VEHICLE_SUB_CATEGORY_MIDSIZE_CARS',
                          'VEHICLE_SUB_CATEGORY_MIDSIZE_SUVS',
                          'VEHICLE_SUB_CATEGORY_MINI_SPORT_UTILITIES', 'VEHICLE_SUB_CATEGORY_MINIVANS',
                          'VEHICLE_SUB_CATEGORY_NEAR_LUXURY', 'VEHICLE_SUB_CATEGORY_OPEN_TOPS',
                          'VEHICLE_SUB_CATEGORY_PASSENGER_VANS', 'VEHICLE_SUB_CATEGORY_SMALL_CARS',
                          'VEHICLE_SUB_CATEGORY_SPECIALTY_TRUCKS', 'VEHICLE_SUB_CATEGORY_SPORTY_CARS',
                          'VEHICLE_SUB_CATEGORY_STATELY_WAGON', 'VEHICLE_SUB_CATEGORY_STATION_WAGONS',
                          'VEHICLE_SUB_CATEGORY_TRUE_LUXURY', 'MAIL_ORDER_RESPONDER_INSURANCE',
                          'COLLEGE_GRAD_TRIGGER', 'EMPTY_NESTER_TRIGGER', 'NEW_FIRST_CHILD_0_2_TRIGGER',
                          'HOME_MKT_VALUE_TRIGGER', 'INCOME_TRIGGER',
                          'NEW_ADULT_TO_FILE_TRIGGER', 'NEW_PRE_DRIVER_TRIGGER', 'NEW_YNG_ADULT_TO_FILE_TRIGGER',
                          'NEW_MARRIED_TRIGGER', 'NEW_SINGLE_TRIGGER', 'RETIRED_TRIGGER', 'VALUESCORE_TRIGGER']

        onedict = {'Y': 1, 'N': 0}

        for variable in variable_list1:
            if variable in data.columns:
                data["{}_Epsilon_Recoded".format(variable)] = data[variable].map(onedict)

        # ANALYSIS: *** Step 2 ***
        # Record MertiScore
        meritdict = {'A1': 12, 'A2': 11, 'B1': 10, 'B2': 9, 'C1': 8, 'C2': 7, 'D1': 6, 'D2': 5, 'D3': 4, 'E1': 3,
                     'E2': 2, 'E3': 1}
        data['MERITSCORE_Epsilon_Recoded'] = data['MERITSCORE'].map(meritdict)

        # Record Letters to Numbers
        valuedict = {'A1': 12, 'A2': 11, 'B1': 10, 'B2': 9, 'C1': 8, 'C2': 7, 'D1': 6, 'D2': 5, 'D3': 4, 'E1': 3,
                     'E2': 2, 'E3': 1}

        data['TARGET_VALUESCORE_20_ALL_MARKETERS_Epsilon_Recoded'] = data['TARGET_VALUESCORE_20_ALL_MARKETERS'].map(
            valuedict)

        data['TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS_Epsilon_Recoded'] = data[
            'TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS'].map(valuedict)

        data['TARGET_VALUESCORE_20_BANK_CARD_MARKETERS_Epsilon_Recoded'] = data[
            'TARGET_VALUESCORE_20_BANK_CARD_MARKETERS'].map(valuedict)

        data['TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS_Epsilon_Recoded'] = data[
            'TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS'].map(valuedict)

        data['TRIGGERVAL_VALUESCORE_Epsilon_Recoded'] = data['TRIGGERVAL_VALUESCORE'].map(valuedict)

        squaredict = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
                      'M': 13, 'N': 14,
                      'O': 15, 'P': 16, 'Q': 17, 'Z': None}

        data['LIVING_AREA_SQ_FTG_RANGE_Epsilon_Recoded'] = data['LIVING_AREA_SQ_FTG_RANGE'].map(squaredict)

        data['TRIGGERVAL_HOME_MKT_VALUE_Epsilon_Recoded'] = data['TRIGGERVAL_HOME_MKT_VALUE'].map(squaredict)

        incomedict = {'A': 10, 'B': 11, 'C': 12, 'D': 13, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3,
                      '2': 2, '1': 1}

        data['TRIGGERVAL_INCOME_Recoded'] = data['TRIGGERVAL_INCOME'].map(incomedict)

        data['ADV_TGT_INCOME_30_Epsilon_Recoded'] = data['ADV_TGT_INCOME_30'].map(incomedict)

        data['PROPERTY_LOT_SIZE_IN_ACRES_Epsilon_Recoded'] = data['PROPERTY_LOT_SIZE_IN_ACRES'].map(incomedict)

        # ANALYSIS: *** Step 2 ***
        data['ADV_TGT_NARROW_BAND_INCOME_30'] = data['ADV_TGT_NARROW_BAND_INCOME_30'].astype(str)

        def bandincome_rewritten_2(data):
            mapping_rules = {'nan': None, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15,
                             'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22,
                             'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32,
                             'X': 33, 'Y': 34, 'Z': 35}
            your_keyword = data
            if your_keyword in mapping_rules.keys():
                return mapping_rules[your_keyword]

        data['ADV_TGT_NARROW_BAND_INCOME_30_Epsilon_Recoded'] = data['ADV_TGT_NARROW_BAND_INCOME_30'].apply(
            lambda x: bandincome_rewritten_2(x))

        # creating list of epsilon recode variable
        epsilon_recoded_list = ['MERITSCORE', 'TARGET_VALUESCORE_20_ALL_MARKETERS',
                                'TARGET_VALUESCORE_20_AUTO_FINANCE_MARKETERS',
                                'TARGET_VALUESCORE_20_BANK_CARD_MARKETERS',
                                'TARGET_VALUESCORE_20_RETAIL_CARD_MARKETERS', 'TRIGGERVAL_VALUESCORE',
                                'LIVING_AREA_SQ_FTG_RANGE',
                                'TRIGGERVAL_HOME_MKT_VALUE', 'TRIGGERVAL_INCOME', 'ADV_TGT_INCOME_30',
                                'PROPERTY_LOT_SIZE_IN_ACRES',
                                'ADV_TGT_NARROW_BAND_INCOME_30']
        epsilon_recoded_list.extend(variable_list1)
        epsilon_recoded_list = [x.lower() for x in epsilon_recoded_list]

        return data, epsilon_recoded_list


class StatisticalTools:

    @staticmethod
    def classify_variable(vec: pd.Series,
                          distint_values_threshold: int = 10,
                          max_values: int = 1000,
                          max_categories: int = 40,
                          missing_threshold: float = 0.95) -> str:
        """
        Def: This function will classify the type/kind of each variable. There are five types of variables:
        'exclude', 'continuous', 'binary', 'categorical', 'categorical+other'( categorical variable exceeds number of
        max_categories)

        :param vec: pd.Series, of the variable column
        :param distint_values_threshold: For numeric variables, if number of values less than distint_values_threshold
        then classify this variable as 'categorical'
        :param max_values: maximum values of a non-numeric variable, if number of values larger than max_val
        then classify as 'exclude'
        :param max_categories: maximum values of a categorical variable, if number of values larger than max_cats
        then 'Categorical+other'
        :param missing_threshold: missing threshold, if the missing value percentage larger than this threshold then
        classify as 'exclude'
        :return: str, the type/kind of the variable
        """
        nvals = vec.nunique()
        if DataTools.isnumeric(vec.unique()):
            is_char_type = False
        else:
            is_char_type = True
        # Checks for zero variance or near zero variance
        if ((vec.value_counts(normalize=True)) * 100 > 95).any():
            return 'exclude'
        elif vec.isna().sum() / len(vec) > missing_threshold:
            return 'exclude'
        else:
            if not is_char_type:
                # check for zero variance or near zero
                if nvals <= distint_values_threshold:
                    if nvals == 2:
                        return 'binary'
                    elif nvals > max_categories:
                        return 'categorical+other'
                    else:
                        return 'categorical'
                else:
                    return 'continuous'
            else:
                if nvals > max_values:
                    return 'exclude'
                elif nvals > max_categories:
                    return 'categorical+other'
                else:
                    return 'categorical'

    @staticmethod
    def classify_variable_v2(vector: pd.Series,
                             missing_exclude_threshold: float = 0.95,
                             default_type: str = 'exclude',
                             categorical_threshold: int = 100,
                             string_threshold: int = 100):
        """
        Def: This function will classify the type/kind of each variable. There are five types of variables:
        'exclude', 'continuous', 'binary', 'categorical', 'categorical+other'( categorical variable exceeds number of
        max_categories)

        :param vector: pd.Series, of the variable column
        :param missing_exclude_threshold: missing threshold, if the missing value percentage larger than this threshold
        then classify as 'exclude'
        :param default_type: str, default variable type
        :param categorical_threshold: maximum values of a categorical variable, if number of values larger than max_cats
        then 'Categorical+other'
        :param string_threshold: variable which is string in nature and number of distinct values is more than this
        threshold then classify as 'exclude'
        :return: str, the type/kind of the variable
        """
        # number of unique value in the column
        unique_values = vector.unique()

        # Normalised counts saved to avoid multiple computations
        normalised_counts = len(vector.value_counts(normalize=True))

        # removing the epsilon variable which has been recoded

        # Zero Variance variable
        if normalised_counts == 1:
            return 'exclude'

        # Near Zero Variance variable
        elif (100 * vector.value_counts(normalize=True) > 95).any():
            return 'exclude'

        # variable having more than specified threshold of missing value
        elif vector.isnull().sum() / len(vector) > missing_exclude_threshold:
            return 'exclude'

        # removing variable which is string in nature and having more than 100 distinct values
        elif pd.api.types.is_string_dtype(vector) and len(unique_values) > string_threshold:
            return 'exclude'

        # variable which have only 2 values
        elif normalised_counts == 2:
            return 'binary'

        # categorical variable
        elif pd.api.types.is_string_dtype(vector) and len(unique_values) <= categorical_threshold:
            return 'categorical'

        # numerical variable
        elif pd.api.types.is_numeric_dtype(vector):
            return 'continuous'

        # in case if function missed to capture any variable, it will ask user what to do with the variables
        else:
            return default_type

    @staticmethod
    def get_breaks(pdSeries, split_type='quantity', nbins=3, squash_extremes=False):
        """
        :param pdSeries: pd.Series of numeric data
        :param split_type: split_type 'width' or 'quantity'. If 'width', breakpoints will be evenly-spaced.
            If 'quantity', breakpoints will be made to create even-quantity bins.
        :param nbins: int of segments to create
        :param squash_extremes: bool to handle extreme values that requiring manual binning
        :return: pd.Series of breakpoints to used in the cut function to split vec into nbins segments
        """
        # remove null values in the column
        # np.sum(pdSeries.isnull())
        not_null_series = pdSeries[pdSeries.notnull()]
        if squash_extremes:
            # likely unncessary since base pd.cut already takes care of this
            if type(nbins) != int:
                not_null_series = pd.Series(np.where(
                    (not_null_series < nbins[0]) | (not_null_series > nbins[-1]), None, not_null_series))
            else:
                raise AttributeError(
                    'not possible to convert extreme values to nulls unless using manual binning'
                )

        # check if numeric column
        if not pd.api.types.is_numeric_dtype(not_null_series):
            not_null_series = not_null_series.astype(float)  # if not try convert to numeric

        if split_type == 'width':
            breaks = pd.cut(not_null_series, nbins, right=False, precision=6)
        else:
            # TODO Check with TongTong about alternative split method
            breaks = pd.qcut(not_null_series, nbins, duplicates='drop')
        # return breaks.value_counts(), pd.Series([pdSeries.isna().sum()], index=[None])
        return breaks

    @staticmethod
    def profile_segment(subset,
                        subset_binned_name: str,
                        variable_name: str,
                        dependent_variable: str = None,
                        dependent_variable_average: str = None,
                        ) -> pd.DataFrame:
        """
        Def: This function is used in create_profile() for generating profile columns for each variable

        :param subset: pd.Series, of the variable
        :param subset_binned_name: str, the name of the variable to be grouped by
        :param variable_name: str, name of the variable
        :param dependent_variable: str, dependent variable
        :param dependent_variable_average:
        :return: pd.DataFrame, return the profiling result dataframe of the variable
        """
        total = len(subset)
        # if blanks_as_na:
        #     subset = subset.fillna('NA')
        final_selection = ['Variable', 'Category', 'Count', 'Percent']
        if not dependent_variable:
            output = subset.groupby(subset_binned_name).size().reset_index(name='counts')
            output['Percent'] = (output['counts'] / total) * 100
            output = output.rename(columns={subset_binned_name: 'Category',
                                            'counts': 'Count'})

        else:
            final_selection.append('index')
            if not dependent_variable_average:
                dependent_variable_average = subset[dependent_variable].mean()
            output = subset.groupby(subset_binned_name).agg({
                subset_binned_name: 'count',
                dependent_variable: 'mean'
            })

            output['Percent'] = (output[subset_binned_name] / total)
            output['index'] = (output[dependent_variable] / dependent_variable_average) * 100
            output = output.rename(columns={subset_binned_name: 'Count'})
            output.rename(columns={dependent_variable: 'mean_DV'}, inplace=True)
            final_selection.append('mean_DV')
            output['Category'] = output.index

        output['Variable'] = variable_name
        output = output.reset_index()
        
        
        return output[final_selection]

    @staticmethod
    def quantile_function(a, bounds):
        '''Quantile based on Nearest Neighbour'''

        bnds_temp = [np.quantile(a, q=(.5 - bounds / 2), interpolation='nearest'),
                     np.quantile(a, q=(.5 + bounds / 2), interpolation='nearest')]

        return bnds_temp

    @staticmethod
    def iv_woe(data, target, bins=10, force_bin=3, show_woe=False):
        """
        Def: information value creation based on number of bins for categorical/binary

        :param data: pd.Dataframe of categorical/binary
        :param target: dependent variable
        :param bins: int, number of bins default to 10
        :param force_bin: **
        :param show_woe: bool, print the result or not
        :return: pd.Dataframe
        """
        # Empty Dataframe
        newDF, woeDF = pd.DataFrame(), pd.DataFrame()

        # Extract Column Names
        cols = data.columns

        # Run WOE and IV on all the independent variables
        for ivars in cols[~cols.isin([target])]:
            # print(ivars)
            if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 20):
                # print("create bins...")
                binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
                d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
                d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
                # print(len(d))
                if (len(d) == 1):
                    bins_t = algos.quantile(data[ivars], np.linspace(0, 1, force_bin))
                    # print(bins_t)
                    if len(np.unique(bins_t)) == 2:
                        ls = [bins_t[0], data[ivars].value_counts().index[1], bins_t[-1]]
                        # print(ls)
                        # TODO to confirm with Sinan
                        d0 = pd.DataFrame(
                            {'x': pd.cut(data[ivars], ls, include_lowest=True, duplicates='drop'), 'y': data[target]})
                    else:
                        # TODO to confirm with Sinan
                        d0 = pd.DataFrame(
                            {'x': pd.cut(data[ivars], np.unique(bins_t), include_lowest=True, duplicates='drop'),
                             'y': data[target]})
                    d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
                else:
                    d = d
            else:
                # print("use original bins")
                d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
                d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})

            d.columns = ['Cutoff', 'N', 'Events']
            d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
            d['Non-Events'] = d['N'] - d['Events']
            d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
            d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
            d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
            d.insert(loc=0, column='Variable', value=ivars)
            # print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
            temp = pd.DataFrame({"variable": [ivars], "inf_val": [d['IV'].sum()]},
                                columns=["variable", "inf_val"])
            newDF = pd.concat([newDF, temp], axis=0)
            woeDF = pd.concat([woeDF, d], axis=0)

            # Show WOE Table
            if show_woe:
                print(d)
        return newDF


class PythonTools:

    @staticmethod
    def get_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
