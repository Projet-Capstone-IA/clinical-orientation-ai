"""
Module de dataset personnalisé pour les données MedQuAD.
Format adapté pour Mistral avec masquage de la partie question.
"""

import torch
from torch.utils.data import Dataset

class MedQADataset(Dataset):
    """
    Dataset PyTorch personnalisé pour les données MedQuAD.
    
    Transforme chaque paire question-réponse au format attendu par Mistral :
    <s>[INST] question [/INST] réponse </s>
    
    Attributes
    ----------
    data : pd.DataFrame
        DataFrame contenant les colonnes 'question' et 'answer'
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer du modèle (Mistral)
    max_length : int
        Longueur maximale de séquence (défaut: 512)
    """
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        """
        Initialise le dataset avec les données et le tokenizer.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame avec colonnes 'question' et 'answer'
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer pour encoder le texte
        max_length : int, optional
            Longueur maximale des séquences (par défaut 512)
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """
        Retourne le nombre total d'exemples dans le dataset.

        Returns
        -------
        int
            Nombre d'exemples
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Récupère un exemple tokenizé à l'index donné.

        Parameters
        ----------
        idx : int
            Index de l'exemple

        Returns
        -------
        dict
            Dictionnaire contenant:
            - input_ids : torch.Tensor
                IDs des tokens (forme: [max_length])
            - attention_mask : torch.Tensor
                Masque d'attention (forme: [max_length])  
            - labels : torch.Tensor
                IDs des tokens avec -100 pour la partie question
        """
        question = self.data.loc[idx, 'question']
        answer = self.data.loc[idx, 'answer']
        
        # Format Mistral: <s>[INST] Question [/INST] Réponse </s>
        text = f"<s>[INST] {question} [/INST] {answer} </s>"
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Pour le calcul de la loss, on masque la partie question
        labels = encoding['input_ids'].clone()
        
        # Trouver où se termine l'instruction pour masquer la question
        inst_end = text.find('[/INST]') + len('[/INST]')
        inst_text = text[:inst_end]
        inst_tokens = self.tokenizer(inst_text, add_special_tokens=False)['input_ids']
        
        # Masquer les tokens de l'instruction (mettre -100)
        labels[0, :len(inst_tokens)] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }