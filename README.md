# Convex Optimization-Based Municipal Parking Lot Recommendation System in Seoul
* `Period: Term project conducted in the Convex Optimization Lecture at KyungHee University, 2st Semester 2025, 2025.09 ~ 2025.12`

<br/><br/>

# ✨ Team K2R1
1. 구성원
* 팀장: Jinman Kim - Ph.D. Stundent (Part-time, [TM Lab](https://tmlab.khu.ac.kr/home)), Dept. of Big Data Analytics, KyungHee University
* 팀원: Hyeonjong Jang - M.S. Stundent ([AIMS Lab](https://sites.google.com/khu.ac.kr/aims/home?authuser=0)), Dept. of Artificial Intelligence, KyungHee University
* 팀원: Jaejoon Choi - M.S. Stundent ([AIMS Lab](https://sites.google.com/khu.ac.kr/aims/home?authuser=0)), Dept. of Industrial and Management Systems Engineering, KyungHee University
* 팀원: Yunseo Hwang - M.S. Stundent ([TM Lab](https://tmlab.khu.ac.kr/home)), Dept. of Industrial and Management Systems Engineering, KyungHee University

<br/>

2. 주 임무
* Jinman Kim - Framework Design, Data Collection and Preprocessing
* Hyeonjong Jang - System Implementation, Prompt Engineering
* Jaejoon Choi - System Implementation, Prompt Engineering
* Yunseo Hwang - Data Collection and Preprocessing, Model Validation

<br/><br/>

# 🗂 Presentation
## 1. Data
<img width="1879" height="755" alt="image" src="https://github.com/user-attachments/assets/e3ca4038-3921-4c51-b0eb-df2cd7864f5d" />

<br/><br/>

<img width="1838" height="798" alt="image" src="https://github.com/user-attachments/assets/86dc7bfd-d00b-4df3-8418-26f63e7434dd" />


<br/><br/>
## 2. Problem Definition
<img width="1948" height="968" alt="image" src="https://github.com/user-attachments/assets/935d164b-1517-4bb5-8ce8-a04f5f089e75" />

<br/><br/>

<img width="1834" height="679" alt="image" src="https://github.com/user-attachments/assets/9ee7eac7-1f3d-49e9-8da2-2777961cd3c3" />


<br/><br/>
## 3. Purpose & Modeling
<img width="1984" height="920" alt="image" src="https://github.com/user-attachments/assets/a43e2546-e293-4aad-bfc1-718d554a2cca" />

<br/><br/>

<img width="1889" height="986" alt="image" src="https://github.com/user-attachments/assets/6a4cba39-239e-4ab1-b6da-35f0d637ad9d" />

<br/><br/>

<img width="1809" height="877" alt="image" src="https://github.com/user-attachments/assets/f959a823-0cbe-4f6a-8a68-e18570b07e42" />


<br/><br/>
## 4. Experiment Design And Results 
<img width="1893" height="861" alt="image" src="https://github.com/user-attachments/assets/f6541b98-66b2-4cc8-a231-11e88749b7ba" />
<img width="723" height="386" alt="image" src="https://github.com/user-attachments/assets/cb0d994c-ee35-498d-baf2-04db19156aca" />

<br/><br/>

<img width="1925" height="980" alt="image" src="https://github.com/user-attachments/assets/99e05179-9adc-41c6-8e07-373f470d158e" />
<img width="673" height="385" alt="image" src="https://github.com/user-attachments/assets/b06b5718-f297-4f83-b619-389a5a7c8e1b" />

<br/><br/>

<img width="2006" height="980" alt="image" src="https://github.com/user-attachments/assets/1fa4e08a-766c-4717-90c0-1ae891dc5133" />
<img width="673" height="385" alt="image" src="https://github.com/user-attachments/assets/b06b5718-f297-4f83-b619-389a5a7c8e1b" />

<br/><br/>
## 5. System UI/UX
<img width="1832" height="909" alt="image" src="https://github.com/user-attachments/assets/39b62297-d89f-4a5e-b9dd-d89bd003a534" />

<br/><br/>

<img width="1904" height="950" alt="image" src="https://github.com/user-attachments/assets/1baf318c-cb30-4461-abcc-acaf04608de4" />

<br/><br/>

<img width="1947" height="734" alt="image" src="https://github.com/user-attachments/assets/62fde055-2942-494b-99ef-c69647b24f6f" />


<br/><br/>
## 6. Conclusion & Discussion
- 상한 제약(ρ_max=0.95)으로 혼잡도가 기존 Greedy 대비 평일 8.2%, 주말 7.5% 개선됨. 
- 고정적인 가중치가 아닌 LLM을 통한 Prompt 기반의 가중치 추천도 성능개선을 보임.
- 평균 거리를 유지하면서 전역 최적해를 보장하지만 상한 제약 추가 시 통계적 유의성이 감소함.
- 강건하고 최적화된 Prompt Engineering이 반영된 가중치 자동 조정 메커니즘의 도입 필요함. 
- 정부 측면에서는 주차 탐색 및 불법 정차 감소와 AI 기반 주차 관리 시스템을 활용할 수 있음.
- 이용자 측면에서는 개인 목적지 맞춤화에 따른 시간 절약과 비용 및 스트레스 감소의 이점이 존재함.
- 알고리즘 및 모델 측면에서는 거리, 혼잡도, 비용을 동시에 고려한 종합적 최적화와 전역 최적성을 보장할 수 있음.

