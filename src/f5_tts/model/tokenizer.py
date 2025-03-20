import re
from typing import List

class VietnameseTokenizer:
    def __init__(self):
        # Định nghĩa các ký tự đặc biệt trong tiếng Việt
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
            '<space>': 4
        }
        
        # Định nghĩa các ký tự tiếng Việt
        self.vietnamese_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?')
        self.vietnamese_tones = set('áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ')
        
        # Tạo từ điển token
        self.token_dict = {**self.special_tokens}
        current_idx = len(self.special_tokens)
        
        # Thêm các ký tự tiếng Việt vào từ điển
        for char in sorted(self.vietnamese_chars | self.vietnamese_tones):
            self.token_dict[char] = current_idx
            current_idx += 1
            
        # Tạo từ điển ngược
        self.reverse_token_dict = {v: k for k, v in self.token_dict.items()}
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize văn bản tiếng Việt thành các token."""
        # Chuẩn hóa văn bản
        text = text.lower()
        
        # Tách các ký tự đặc biệt
        text = re.sub(r'([.,!?])', r' \1 ', text)
        
        # Tách các từ
        tokens = []
        current_token = ''
        
        for char in text:
            if char in self.vietnamese_chars or char in self.vietnamese_tones:
                current_token += char
            elif char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                tokens.append('<space>')
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                if char in self.special_tokens:
                    tokens.append(char)
                    
        if current_token:
            tokens.append(current_token)
            
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Chuyển đổi văn bản thành các token ID."""
        tokens = self.tokenize(text)
        return [self.token_dict.get(token, self.token_dict['<unk>']) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Chuyển đổi các token ID thành văn bản."""
        tokens = [self.reverse_token_dict.get(id, '<unk>') for id in token_ids]
        text = ''.join(tokens)
        # Khôi phục khoảng trắng
        text = text.replace('<space>', ' ')
        return text
    
    def vocab_size(self) -> int:
        """Trả về kích thước từ vựng."""
        return len(self.token_dict) 