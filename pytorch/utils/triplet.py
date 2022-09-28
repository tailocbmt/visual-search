from abc import abstractmethod
import pandas as pd
import random

from pytorch.utils.params import CATEGORY_ID, STYLE, IMAGE_NAME, X1, X2, Y1, Y2, PAIR_ID


class TripletBaseClass:
    def __init__(self, triplet_per_image: int, dataframe: pd.DataFrame) -> None:
        self.NUM_LABELS: int = 13
        self.EXCLUDE_STYLE: int = 0
        self.POSTFIXES = ['_a', '_p', '_n']
        self.TAKEN_COLUMNS = [IMAGE_NAME, CATEGORY_ID, STYLE, X1, Y1, X2, Y2]
        self.SAVED_COLUMNS = [column+postfix for postfix in self.POSTFIXES for column in self.TAKEN_COLUMNS]

        self.triplet_per_image = triplet_per_image
        self.dataframe = dataframe
    
    @abstractmethod
    def init_data(self) -> None:
        pass
    
    @abstractmethod
    def create_triplets(self) -> pd.DataFrame:
        pass

    def run(self, save_path: str='') -> None:
        self.init_data()
        dataframe = self.create_triplets()

        if save_path:
            dataframe.to_csv(save_path, index=False)


class TripletWithSameCategoryNegative(TripletBaseClass):
    def __init__(self, triplet_per_image: int, dataframe: pd.DataFrame) -> None:
        super().__init__(triplet_per_image, dataframe)
    
    def init_data(self) -> None:
        # Create ids list with same category id
        self.same_category_ids = [[]]
        for i in range(1, 14):
            same_category = self.dataframe[self.dataframe[CATEGORY_ID] == i].index
            self.same_category_ids.append(same_category)

    def create_triplets(self) -> pd.DataFrame:
        results = []
        df_grouby_pair_id = self.dataframe.groupby(by=[PAIR_ID])

        for _, data in df_grouby_pair_id:
            for anchor_index, anchor_row in data.iterrows():
                if anchor_row == self.EXCLUDE_STYLE:
                    continue

                anchor_style = anchor_row[STYLE]
                anchor_category_id = anchor_row[CATEGORY_ID]
                anchor_data = anchor_row[self.TAKEN_COLUMNS]

                ids_with_same_style = list(data[(data[CATEGORY_ID] == anchor_category_id) & (data[STYLE] == anchor_style)].index)
                data_with_same_style = self.dataframe.loc[ids_with_same_style, self.TAKEN_COLUMNS]

                ids_with_same_category = self.same_category_ids[anchor_category_id]
                ids_with_same_category = [index for index in ids_with_same_category if index not in ids_with_same_style]
                ids_with_same_style.remove(anchor_index)
                
                for _, positive_row in data_with_same_style.iterrows():
                    positive_data = positive_row[self.TAKEN_COLUMNS].values.tolist()
                    ids_negative = random.choices(ids_with_same_category, k=self.triplet_per_image)
                    data_with_negative_ids = self.dataframe[ids_negative, self.TAKEN_COLUMNS]

                    for _,negative_row in data_with_negative_ids.iterrows():
                        triplet = []
                        negative_data = negative_row[self.TAKEN_COLUMNS].values.tolist()

                        triplet.extend(anchor_data)
                        triplet.extend(positive_data)
                        triplet.extend(negative_data)
                        results.append(triplet)
        
        result_df = pd.DataFrame(results, columns=self.SAVED_COLUMNS)
        return result_df


class TripletWithSamePairIdThenSameCategory(TripletWithSameCategoryNegative):
    def __init__(self, triplet_per_image: int, dataframe: pd.DataFrame) -> None:
        super().__init__(triplet_per_image, dataframe)
    
    def init_data(self) -> None:
        return super().init_data()
    
    def create_triplets(self) -> pd.DataFrame:
        results = []
        df_grouby_pair_id = self.dataframe.groupby(by=[PAIR_ID])

        for _, data in df_grouby_pair_id:
            for anchor_index, anchor_row in data.iterrows():
                if anchor_row == self.EXCLUDE_STYLE:
                    continue

                anchor_style = anchor_row[STYLE]
                anchor_category_id = anchor_row[CATEGORY_ID]
                anchor_data = anchor_row[self.TAKEN_COLUMNS]

                ids_with_same_style = list(data[(data[CATEGORY_ID] == anchor_category_id) & (data[STYLE] == anchor_style)].index)
                data_with_same_style = self.dataframe.loc[ids_with_same_style, self.TAKEN_COLUMNS]

                # Choose in pair id, same category but different style first
                ids_with_same_category = list(data[(data[CATEGORY_ID] == anchor_category_id) & (data[STYLE] != anchor_style)].index)
                
                # elso choose in same category 
                if len(ids_with_same_category) == 0:
                    ids_with_same_category = self.same_category_ids[anchor_category_id]
                    ids_with_same_category = [index for index in ids_with_same_category if index not in ids_with_same_style]

                ids_with_same_style.remove(anchor_index)
                
                for _, positive_row in data_with_same_style.iterrows():
                    positive_data = positive_row[self.TAKEN_COLUMNS].values.tolist()
                    ids_negative = random.choices(ids_with_same_category, k=self.triplet_per_image)
                    data_with_negative_ids = self.dataframe[ids_negative, self.TAKEN_COLUMNS]

                    for _,negative_row in data_with_negative_ids.iterrows():
                        triplet = []
                        negative_data = negative_row[self.TAKEN_COLUMNS].values.tolist()

                        triplet.extend(anchor_data)
                        triplet.extend(positive_data)
                        triplet.extend(negative_data)
                        results.append(triplet)
        
        result_df = pd.DataFrame(results, columns=self.SAVED_COLUMNS)
        return result_df