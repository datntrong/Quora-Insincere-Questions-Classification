# Quora-Insincere-Questions-Classification
# **Quora Insincere Questions Classification**

# Mô tả bài toán

## Giới thiệu bài toán

1. Quora là một nền tảng cho phép mọi người học hỏi lẫn nhau. Trên Quora, mọi người có thể đặt câu hỏi và kết nối với những người khác, những người đóng góp thông tin chi tiết độc đáo và câu trả lời chất lượng.
2. Một vấn đề tồn tại đối với bất kỳ trang web lớn nào hiện nay là làm thế nào để xử lý nội dung độc hại và gây chia rẽ.
3. Quora muốn giải quyết vấn đề này trực tiếp để giữ cho nền tảng của họ trở thành một nơi mà người dùng có thể cảm thấy an toàn khi chia sẻ kiến thức của họ với thế giới.
4. Một thách thức quan trọng là loại bỏ những câu hỏi thiếu chân thành - những câu hỏi được đặt ra dựa trên những tiền đề sai lầm hoặc có ý định đưa ra một tuyên bố hơn là tìm kiếm những câu trả lời hữu ích.
5. Trong báo cáo,em đề xuất phát triển các mô hình xác định và gắn cờ cho các "insincere questions"

## Input/Output

Input : Tập train và test.

Tập train : Gồm 3 trường dữ liệu : 

- qid : id của câu hỏi
- question_text : Nội dung câu hỏi
- target : Nhãn của câu hỏi phân loại câu hỏi bằng hai giá trị là 0, 1 với 0 là các câu hỏi "**sincere**" và 1 là các câu hỏi "**insincere**"
- Có 1225312 nhãn 0, 80810 nhãn 1

Tập test : Gồm 2 trường dữ liệu : 

- qid : id của câu hỏi
- question_text : Nội dung câu hỏi

Chúng ta chỉ quan tâm dữ liệu về tập train.

- Tổng quan về dữ liệu train
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/555dbe72-1a49-436a-b484-fccbcf2d1ac1/Untitled.png)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06afdef7-dfa8-4385-82c2-3aeb7b646c8b/Untitled.png)
    

## **Visualize data**

### **Thống kê về dữ liệu**

Qua một vài thống kê, em nhận thấy **"sincere question"** có hơn 1,2 triệu câu hỏi, chiếm 93.8% tập train, còn lại 6.2% cho **"insincere question"** với khoảng gần 81,000 câu hỏi. Nhận thấy lượng data có sự chênh lệnh lớn khi số lượng **"sincere question"** gấp 15 lần so với **"insincere question"**. Với lượng data bị mất cân bằng giữa các nhãn như thế này, chúng ta cần phải có một biện pháp khắc phục sự mất cân bằng để mô hình cho được kết quả tốt nhất.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cb04f550-a94e-484d-ab22-be636478a261/Untitled.png)

### **Phân tích câu trong data**
Chúng ta có thể nhận ra : Các câu trong data có độ dài không giống nhau, hầu hết tập trung từ khoảng 0-30 từ, vẫn có các câu có độ dài lớn. Ở đây em chọn 70 làm maxlen, tức là độ dài câu mà chúng ta sẽ sử dụng để biểu diễn trong ma trận , lúc này tất cả các câu hỏi sẽ có cùng độ dài, đạt tiêu chuẩn đầu vào của mô hình train.

## **Tiền xử lý dữ liệu**

### **Làm sạch dữ liệu**

Ở bước làm sạch dữ liệu này em xây dựng 3 hàm :

- **clean_text** :xử lý các ký tự đặc biệt tồn tại trong văn bản đầu vào, các ký tự đặc biệt được lấy từ mảng puncts bên dưới. Hiểu đơn giản là bỏ đi các ký tự đặc biệt và xóa dấu câu
- **clean_number** : Xử lý các chữ số đầu vào, thay thế bằng các ký tự #s tại vì các thư viện embedding đã xử lý các số theo cách này
- **replace_typical_misspell**: xử lý các từ viết tắt trong câu đầu vào bằng **mispell_dict** tương ứng(**Removing Contractions)**

### Chuẩn hoá dữ liệu

1. Chuyển hết tất cả các ký tự trong câu hỏi về dạng viết thường
2. Xử các ký tự đặc biệt khỏi từng câu : Các kí tự như “:,:))))),...” không có giá trị gì cho phân tích văn bản nên sẽ loại bỏ
3. Xử lý các chữ số nằm trong từng câu hỏi.
4. Thay thế các từ viết tắt thành dạng nguyên bản của chúng
5. Thay thế các giá trị null trong cột questiontext bằng giá trị “##”
6. Thêm một số trường dữ liệu cho dataset.
7. Tokenize train data và test data : Chuyển text -> ma trận từ (số hoá text)
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/489c77dc-2885-4897-9932-8a9015d55ce4/Untitled.png)
    
8. Padding data : Lấp đầy nhưng câu ngắn bằng những số 0.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/715182cf-d83e-44ec-904b-0d334dbe2ab7/Untitled.png)
    
9. Shuffle data: Trộn data một cách ngẫu nhiên

### **Embedding**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e4e415e0-6c2e-435d-9027-ac83e9e7b280/Untitled.png)

Chúng ta phải chuyển đổi từ text qua ma trận để máy tính có thể hiểu được. Có nhiều phương pháp để biểu diễn như one-hot, TF-IDF,... Tuy nhiên trong cuộc thi này, bộ data của Quora Insincere Questions có chứa luôn 3 thư viện về embeding lớn nhất **glove, wiki-news (fasttext), paragram** và **GoogleNews** nên em sẽ sử dụng luôn bộ dữ liệu embedding này.

Ban đầu, mỗi lần chạy mô hình em đều phải tiền xử lý lại data 1 lần, điều đó rất mất thời gian đợi cũng như giới hạn thời gian chạy bằng GPU của Kaggle, nên em đã lưu lại dữ liệu tiền xử lý và xuất nó ra làm 1 file data riêng để sử dụng cho việc thử nghiệm nhiều mô hình khác nhau, giúp chọn lựa được mô hình tốt nhất. Chỉ model nào tốt nhất được em thử nghiệm và chọn lựa em mới tiến hành đủ, còn những model nào em đang thử nghiệm mà chưa đủ tốt, em xin phép được chạy bằng data đã tiền xử lý để tiết kiệm thời gian.

Toàn bộ code của quá trình tiền xử lý được em mô tả rõ ràng trong file sau:

# Mô hình

## **Cross Validation**

Bộ dataset chúng ta chỉ có 2 tập, train và test, không có tập val. Chúng ta cũng không được dùng tập train để kiểm thử mô hình, vì nó sẽ dẫn tới overfitting trên tập train. Vậy chúng ta sẽ lấy 1 phần của tập train ra làm tập validation. Nhưng, tập train của dữ liệu quá ít nhãn 1, việc lấy ra 1 phần của tập lỡ như hầu hết nhãn 1 đều nằm trong tập val này thì dữ liệu nhãn 1 ở tập train sẽ ít đi và dẫn đến thiếu dữ liệu train. Điều này dẫn đến mô hình không học được nhãn 1, có thể model không tốt . Để mô hình được huấn luyện tốt nhất, em đề xuất sử dụng Cross Validation.

- Cross Validation là phương pháp chia nhỏ tập training ra thành N phần. Với mỗi lần train, mô hình sẽ sử dụng N-1 phần để train, sau đó val dựa trên 1 phần còn lại. Điều này sẽ giúp cho mô hình hạn chế gặp phải overfitting và giúp bạn tìm ra được những Hyper parameter tốt hơn.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f7b00720-9160-488d-a4ca-be310ac810b5/Untitled.png)

- Như hình chúng ta có thể thấy : Với mỗi lần train đầu, lấy 4 fold đầu tiên để train. Sau đó để val, sử dụng fold 5 để val. Qua lần train thứ 2, bạn lấy từ fold 2 đến fold 5 để train, rồi lại lấy fold 1 để val. Và đó, chính là Cross Validation.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/08ba1fde-d256-4ae8-93b1-85dced20823d/Untitled.png)

- Phương pháp đánh giá : Training data ta chia thành K phần. Sau đó train model K lần, mỗi lần train sẽ chọn 1 phần làm dữ liệu validation và K-1 phần còn lại làm dữ liệu training. Kết quả đánh giá model cuối cùng là trung bình cộng kết quả đánh giá của K lần train.

Em chia dữ liệu thành 5 phần:
Tất cả các mô hình sau em đều sử dụng Cross Validation để huấn luyện.

## CNN Model

Tham khảo tại :

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

Kiến trúc mô hình CNN ( *Source: Zhang, 2015)*

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9418ebb7-dadc-43f2-98db-f2421942447b/Untitled.png)

### Giải thích về mô hình

Đối với ảnh: Chúng ta thường sử dụng các max pooling các điểm ảnh lại với nhau( thường là các ô vuông) ví dụ  9*9 → 4*4. Tức chúng ta kéo các điểm ảnh lại gần hơn để trích xuất đặc trưng tương ứng của vùng ảnh đó. Nhưng đối với text chúng ta không làm như vậy được

Ở đây í tưởng chính của tác giả : Cố gắng học đặc trưng của 1-2-4-8 từ liền kề nhau trong câu, tại vì trong văn bản, các từ thường có quan hệ mật thiết với nhau và từ những đặc trưng đó sẽ phân lớp văn bản. 

Giả sử như hình vẽ trên :

Đầu vào của mô hình là 7 tokens , mỗi token có kích thước vecto embedding là 5 chiều. Kích thước ma trận input là : $7 \times 5$(giống như ma trận ảnh)

Ta có đầu vào : $\mathbf x_{emd} \in \mathbb R^{7\times5}$

Sau đó đầu vào sẽ đi qua một lớp tính chập (1-dimesional convolution) để trích xuất các đặc điểm từ câu. Ví dụ trên, chúng ta có tổng 6 filter, mỗi filter có shape$(f_i,d)$ trong đó $f_i$ là filter size for $i \in \{{1, 2, ..., 6}\}$. Sau đó mỗi filter sẽ scan qua $x_{emd}$ và trả về a feature map:

$$
\mathbf x_{conv_i} = ConV1D(\mathbf x_{emb}) \in \mathbb R^{N-f_i+1}
$$

Tiếp theo dùng hàm kích hoạt ReLU cho $\mathbf x_{conv_i}$ và sử dụng max-over-time-pooling để giảm mỗi feature map to a single scalar. Sau đó chúng ta nối các scalars thành một vecto. Sau đó chúng ta sẽ cho vec tơ này đi qua 1 bộ phân lớp MLP để cho ra output cuối cùng.

$$
\mathbf x_{pool_i} = MaxPool(ReLU(\mathbf x_{conv_i})) \in \mathbb R
$$

$$
\mathbf x_{fc} = concat(\mathbf x_{pool_i}) \in R^{6=size filter}
$$

Cuối cùng, chúng ta cho qua một lớp MLP có ma trận trọng số $\mathbb W_{fc} \in \mathbb R^{2 \times 6}$

$$
logits = Dropout(\mathbb W_{fc} \mathbf x_{fc}) \in \mathbb R^2
$$

- Code của mô hình
    
    ```python
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            
            filter_sizes = [1,2,3,5]
            num_filters = 36
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False
            self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embedding_dim)) for K in filter_sizes])
            self.dropout = nn.Dropout(0.1)
            self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)
    
    #     
        def forward(self, x):
            
            embeds = self.embedding(x[0])
            x = embeds.unsqueeze(1)  
            x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs1]
            x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
            x = torch.cat(x, 1)
            x = self.dropout(x)  
            out = self.fc1(x)
            return out
    ```
    

Giải thích code : 

Ở đây e dùng luôn lớp tích chập 2D nhằm tối ưu tốc độ tính toán.

 num_filters e sử dụng là 36 (vừa đủ để thời gian chạy giảm xuống)

filter_sizes = [1,2,3,5] chúng ta đang xem xét context window( đặc trưng) của 1,2,3,5 từ liên tiếp trong câu. Các cửa sổ filter sẽ dịch từ trên xuống dưới để học các đặc trưng này qua từng lớp tích chập Conv2d.

Source code:

[CNNModel](https://www.kaggle.com/datntrong/cnnmodel)

Huấn luyện và kết quả thực nghiệm : 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0e00b060-fd02-4712-a469-b96875a537a2/Untitled.png)

## BiLSTMModel

- Code của mô hình :
    
    ```python
    class BiLSTM(nn.Module):
        def __init__(self):
            super(BiLSTM, self).__init__()
            self.hidden_size = 64
            drp = 0.1
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False
            self.lstm = nn.LSTM(embedding_dim, self.hidden_size, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(self.hidden_size*4 , 64)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(drp)
            self.out = nn.Linear(64, tagset_size)
        def forward(self, x):
            #rint(x.size())*
            h_embedding = self.embedding(x)
            #_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))*
            h_lstm, _ = self.lstm(h_embedding)
            avg_pool = torch.mean(h_lstm, 1)
            max_pool, _ = torch.max(h_lstm, 1)
            conc = torch.cat(( avg_pool, max_pool), 1)
            conc = self.relu(self.linear(conc))
            conc = self.dropout(conc)
            out = self.out(conc)
            return out
    ```
    

### Mô hình

- Mô hình gồm t input, các input được đưa vào mô hình đúng với thứ tự từ trong câu
- Mỗi hình vuông được gọi là 1 state, đầu vào mỗi state là $x_t$, $h_{t-1}$ với $h_t = f(Wx_t + Uh_{t-1})$. ($\mathbf W$ là trọng số của đầu vào, $\mathbf U$ là trọng số của trạng thái ẩn), $\mathbb f$ là activation value như: sigmoid, tanh, ReLU,....
- Có thể thấy $h_t$ mang cả thông tin từ hidden state trước
- $h_0$ được thêm vào để cho chuẩn công thức nên thường được gán bằng 0 hoặc giá trị ngẫu nhiên
- $y_t = g(V*h_t)$. V là trọng số của trạng thái ẩn sau khi tính đầu ra

### LSTM (Long short term memory)

- Mạng RNN có yếu điểm là không mô tả học được chuỗi quá dài do hiện tượng triệt tiêu đạo hàm (vanishing gradient). Mạng LSTM ra đời khắc phục phần nào nhược điểm này bằng cách cho phép thông tin lan truyền trực tiếp hơn thông qua một biến trạng thái ô (cell state).
- Mạng bộ nhớ dài-ngắn (Long Short Term Memory networks), thường được gọi là LSTM - là một dạng đặc biệt của RNN (Recurrent Neural Network), nó có khả năng học được các phụ thuộc xa. Chúng hoạt động cực kì hiệu quả trên nhiều bài toán khác nhau nên dần đã trở nên phổ biến như hiện nay.
- LSTM được thiết kế để tránh được vấn đề phụ thuộc xa (long-term dependency). Việc nhớ thông tin trong suốt thời gian dài là đặc tính mặc định của chúng, chứ ta không cần phải huấn luyện nó để có thể nhớ được. Tức là ngay nội tại của nó đã có thể ghi nhớ được mà không cần bất kì can thiệp nào.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6e2c18e7-cfe3-414c-af05-eb8366450115/Untitled.png)

- Tại state t
    - output: $c_t$ là cell state, $h_t$ là hidden state
    - input: $c_{t-1},h_{t-1}$. Ở đây $c$ là điểm mới so với RNN
- Tính toán trong cell LSTM:
    - Cổng quên (forget gate): $\mathbf f_t = \sigma(\mathbf W_{f} \mathbf x_t + \mathbf U_{f}\mathbf h_{t-1})$
    - Cổng đầu vào (input gate): $\mathbf i_t = \sigma(\mathbf W_{i} \mathbf x_t + \mathbf U_{i}\mathbf h_{t-1})$
    - Cổng đầu ra (output gate): $\mathbf o_t = \sigma(\mathbf W_{o} \mathbf x_t + \mathbf U_{o}\mathbf h_{t-1})$
    - $\tilde{\mathbf c}t = \mathrm{tanh}(\mathbf W{c} \mathbf x_t + \mathbf U_{c}\mathbf h_{t-1})$
    - Cổng trạng thái ô (cell state): $\mathbf c_{t} = \mathbf f_t \times \mathbf c_{t-1} + \mathbf i_t \times \tilde{\mathbf c}_t$. Forget gate quyết định xem lấy bao nhiêu từ cell state trước và input gate sẽ quyết định lấy bao nhiêu từ input của state và hidden state của state trước
    - $\mathbf h_t = \mathrm{tanh}(c_t) \times \mathbf o_t$  ,
- Chìa khóa của LSTM là trạng thái tế bào (cell state) - chính đường chạy thông ngang phía trên của sơ đồ hình vẽ.Trạng thái tế bào là một dạng giống như băng truyền. Nó chạy xuyên suốt tất cả các mắt xích (các nút mạng) và chỉ tương tác tuyến tính đôi chút. Vì vậy mà các thông tin có thể dễ dàng truyền đi thông suốt mà không sợ bị thay đổi. $c_t$ sẽ được hiệu chỉnh để học được chuỗi dài hơn
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/811927ad-139c-4456-8bb2-7fc2bf7478bb/Untitled.png)
    
- LSTM có khả năng bỏ đi hoặc thêm vào các thông tin cần thiết cho trạng thái tế báo, chúng được điều chỉnh cẩn thận bởi các nhóm được gọi là cổng (gate). Và đó là những khái niệm cốt lõi về LSTM.

### Bidirectional LSTM

- Bidirectional LSTM là mô hình gồm hai LSTM: LSTM thứ nhất nhận đầu vào là chuỗi các từ theo thứ tự từ trái sang phải, LSTM còn lại nhận đầu vào là chuỗi các từ theo thứ tự từ phải sang trái. Cải thiện ngữ cảnh của mô hình, giúp mô hình học được tốt hơn
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3109fa98-3cc4-4697-ba86-9ed2bd5aa05f/Untitled.png)
    

### Kiến trúc mô hình sử dụng

- Lớp Embedding layer
- BiLSTM
- 2 lớp linear, với hàm kích hoạt relu
- Dropout

Toàn bộ code của BiLSTM tại notebook BiLSTMModel + Attention version 15, vì để cải tiến mô hình LSTM tốt hơn với một số kĩ thuật mới, nên em đã copy version 15 sang notebook mới là BiLSTM dưới đây:

Huấn luyện và kết quả thực nghiệm.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cc909787-42b0-4a48-8f8f-9eea5018fbd6/Untitled.png)

## BiLSTMModel + Attention

Tham khảo:

[Attention-Based Deep Learning Model for Aspect Classification on Vietnamese E-commerce Data](https://ieeexplore.ieee.org/abstract/document/9648690)

Em có cài đặt được mô hình này, nhưng chưa thật sự hiểu sâu về nó, nên em xin phép không đề cập vào đây ạ

## BERT

- Tổng quan về BERT
    
    Cuối năm 2018, các nhà nghiên cứu làm việc tại Google AI giới thiệu một một kiến trúc mới cho lớp bài toán Language Representation: BERT - viết tắt của Bidirectional Encoder Representations from Transformers. Đây là mô hình được huấn luyện trước từ các văn bản không được gán nhãn.
    
    Điểm khác biệt lớn nhất so với các mô hình đã có trước đó là mô hình BERT mới này được thiết kế để biểu diễn ngôn ngữ bằng cách xem xét ngữ cảnh 2 chiều cả bên trái và bên phải.  Sau đó, vector biểu diễn ngôn ngữ có thể được tinh chỉnh (bằng một lớp đầu ra bổ sung), ứng dụng cho một loạt các nhiệm vụ, đặc biệt không yêu cầu sửa đổi quá nhiều kiến trúc để phục vụ cho bài toán mới. Điều này góp phần cải thiện hiệu quả khi sử dụng biểu diễn ngôn ngữ được huấn luyện trước cho các tác vụ về sau theo hướng tiếp cận fine-tuning. 
    
    ### Các đóng góp chính của nhóm tác giả trong bài báo
    
    - Chứng minh tầm quan trọng của việc đào tạo trước mô hình biểu diễn ngôn ngữ xem xét ngữ cảnh cả hai chiều.
    - Chứng minh rằng các biểu diễn ngôn ngữ được huấn luyện sẵn giúp mô hình ở các nhiệm vụ sử dụng chúng không cần nhiều kiến trúc phức tạp, cần thiết kế chuyên biệt (đặc thù theo nhiệm vụ cụ thể).
    - Đưa ra mô hình BERT là mô hình đầu tiên (theo hướng tiếp cận fine-tuning) đạt được thành tựu vượt qua nhiều kiến trúc thiết kế đặc thù trong nhiều tác vụ sentence-level và token-level, cải thiện kết quả cho 11 tác vụ xử lý ngôn ngữ tự nhiên.
- Tổng quan về RoBERTa
    - **RoBERTa: A Robustly Optimized BERT Pretraining Approach**
    
    [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
    
    Năm 2019, nhóm tác giả đến từ Facebook AI và trường đại học Washington đã trình bày một nghiên cứu replication của pre-train model BERT và quan sát cẩn thận tác động của việc thay đổi các siêu tham số hay kích thước dữ liệu tới kết quả đào tạo. Bài báo đề xuất một công thức đã được chứng minh để đào tạo các mô hình BERT - RoBERTa - xấp xỉ và có thể vượt qua hiệu quả của các phương pháp post-BERT. Các sửa đổi được áp dụng bao gồm: kéo dài thời gian huấn luyện với kích thước batch lớn hơn và nhiều dữ liệu hơn, không sử dụng bài toán huấn luyện mục tiêu là Dự đoán câu nữa, sử dụng tập huấn luyện có độ dài câu dài hơn, thay đổi động masking pattern được áp dụng cho dữ liệu huấn luyện (thay vì chỉ mask một lần trước giai đoạn train). Kết quả thí nghiệm đạt được đã cho thấy sự cải thiện hơn kết quả của BERT trên cả GLUE và SQuAD. 
    
    Đóng góp của nhóm tác giả trong bài báo này bao gồm:
    
    - Trình bày một tập hợp các lựa chọn thiết kế BERT quan trọng và các chiến lược đào tạo, đồng thời giới thiệu các giải pháp thay thế dẫn đến việc thực hiện nhiệm vụ về sau tốt hơn
    - Sử dụng tập dữ liệu mới, CCNEWS và xác nhận rằng việc sử dụng nhiều dữ liệu hơn để đào tạo trước sẽ cải thiện hơn nữa hiệu suất trên các tác vụ về sau
    - Các cải tiến đào tạo cho thấy rằng masked language model pretraining, theo các lựa chọn thiết kế phù hợp, có tính cạnh tranh với tất cả các phương pháp được công bố khác lúc bấy giờ.

Tutorial về BERT em tham khảo : Tutorials về BERT của thầy Phi

[BERT](https://www.notion.so/BERT-79252528995d4d0f81effc943ce6d6ab) 

- Trích dẫn nguồn tham khảo :
    
    [Google Colaboratory](https://colab.research.google.com/drive/1wdt7z8UcDla3EAjJXI-pxmqZEEB3ry4Z?usp=sharing)
    
    [Google Colaboratory](https://colab.research.google.com/github/huggingface/notebooks/blob/master/transformers_doc/training.ipynb#scrollTo=MZOzuOrK9By8)
    

Code đã train được : 

RoBERTa :

Huấn luyện và kết quả thực nghiệm:

Vì BRET đòi hỏi hỏi kết nối internet là tải pre-train model đã huấn luyện sẵn, nên kết quả không thể submit lên Kaggle được, khi bài toán yêu cầu xử lý trên kernel (local). Việc cài đặt và train mô hình BERT, em chỉ đang học hỏi thêm các phương pháp học sâu để  phân lớp văn bản.

Kết quả mô hình BERT sau 1 epoch :

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5d387ea2-140c-43db-9de1-0d1a2dca0293/Untitled.png)

Em đã chia 80% tập train ra để train và 20% còn lại để val. Kết quả trên tập val

Cho thấy nhãn 1 đạt f1-score lên đến 0.7

# Nhận xét về các mô hình.

## CNN:

Em có chạy thử LR trước đó, và em nhận thấy mô hình CNN cho kết quả khá tốt (tốt hơn LR), nhưng thời gian chạy quá lâu do tính các tích chập. 

## BiLSTM :

Thời gian chạy nhanh, kết quả tốt nhất, mô hình BiLSTM tốt trên dữ liệu dạng text, có một số phương pháp cải thiện tốc độ của BiLSTM, cải thiện điểm, một số phương pháp như Attention, GRU, CLR. Tuy nhiên em vẫn chưa cài đặt được các mô hình này, vì em chưa hiểu hết. Trong tương lai e sẽ tiếp tục cải thiện nó.

## BERT:

Bước đột phá của xử lý ngôn ngữ tự nhiên, kết quả tốt nhất, nhưng thời gian train hơi lâu.