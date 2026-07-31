import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D

class PointCloudOptimizer:
    def __init__(self, cloud1, cloud2, min_distance=0.1):
        """
        포인트 클라우드 최적화 클래스
        
        Parameters:
        cloud1, cloud2: numpy arrays (N x 3) - 3D 포인트 클라우드
        min_distance: float - 최소 허용 거리
        """
        self.cloud1 = np.array(cloud1)
        self.cloud2 = np.array(cloud2)
        self.min_distance = min_distance
        
    def transform_cloud(self, cloud, translation, rotation):
        """포인트 클라우드를 변환 (회전 + 이동)"""
        # 회전 행렬 생성 (오일러 각도 사용)
        rx, ry, rz = rotation
        
        # X축 회전
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        # Y축 회전
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        # Z축 회전
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        # 전체 회전 행렬
        R = Rz @ Ry @ Rx
        
        # 회전 후 이동
        transformed = (R @ cloud.T).T + translation
        return transformed
    
    def compute_closest_distance(self, cloud1, cloud2):
        """두 포인트 클라우드 간 최소 거리 계산"""
        tree = KDTree(cloud2)
        distances, _ = tree.query(cloud1)
        return np.min(distances)
    
    def compute_penalty(self, cloud1, cloud2):
        """충돌 페널티 계산"""
        tree = KDTree(cloud2)
        distances, _ = tree.query(cloud1)
        
        # 최소 거리보다 가까운 점들에 대한 페널티
        violations = distances < self.min_distance
        if np.any(violations):
            penalty = np.sum((self.min_distance - distances[violations])**2)
            return penalty * 1000  # 큰 페널티
        return 0
    
    def objective_function(self, params):
        """목적 함수: 거리 최소화 + 충돌 방지"""
        translation = params[:3]
        rotation = params[3:6]
        
        # cloud2 변환
        transformed_cloud2 = self.transform_cloud(self.cloud2, translation, rotation)
        
        # 최소 거리 계산
        min_dist = self.compute_closest_distance(self.cloud1, transformed_cloud2)
        
        # 충돌 페널티 계산
        penalty = self.compute_penalty(self.cloud1, transformed_cloud2)
        
        # 목적 함수: 거리를 최소화하되 충돌 시 큰 페널티
        return -min_dist + penalty
    
    def optimize(self, initial_guess=None, method='L-BFGS-B'):
        """최적화 실행"""
        if initial_guess is None:
            # 기본 초기값: cloud1의 중심에서 적당히 떨어진 위치
            center1 = np.mean(self.cloud1, axis=0)
            center2 = np.mean(self.cloud2, axis=0)
            offset = center1 - center2 + np.array([2.0, 0.0, 0.0])
            initial_guess = np.concatenate([offset, [0, 0, 0]])
        
        # 최적화 실행
        result = minimize(
            self.objective_function, 
            initial_guess, 
            method=method,
            options={'maxiter': 1000}
        )
        
        return result
    
    def visualize(self, result=None):
        """결과 시각화"""
        fig = plt.figure(figsize=(15, 5))
        
        # 원본 상태
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(self.cloud1[:, 0], self.cloud1[:, 1], self.cloud1[:, 2], 
                   c='red', alpha=0.6, label='Cloud 1')
        ax1.scatter(self.cloud2[:, 0], self.cloud2[:, 1], self.cloud2[:, 2], 
                   c='blue', alpha=0.6, label='Cloud 2')
        ax1.set_title('Original Position')
        ax1.legend()
        
        if result is not None:
            # 최적화 후 상태
            translation = result.x[:3]
            rotation = result.x[3:6]
            transformed_cloud2 = self.transform_cloud(self.cloud2, translation, rotation)
            
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(self.cloud1[:, 0], self.cloud1[:, 1], self.cloud1[:, 2], 
                       c='red', alpha=0.6, label='Cloud 1')
            ax2.scatter(transformed_cloud2[:, 0], transformed_cloud2[:, 1], transformed_cloud2[:, 2], 
                       c='blue', alpha=0.6, label='Cloud 2 (Optimized)')
            ax2.set_title('Optimized Position')
            ax2.legend()
            
            # 최소 거리 계산
            min_dist = self.compute_closest_distance(self.cloud1, transformed_cloud2)
            ax2.text2D(0.05, 0.95, f'Min Distance: {min_dist:.3f}', transform=ax2.transAxes)
            
            # 2D 투영 (XY 평면)
            ax3 = fig.add_subplot(133)
            ax3.scatter(self.cloud1[:, 0], self.cloud1[:, 1], 
                       c='red', alpha=0.6, label='Cloud 1')
            ax3.scatter(transformed_cloud2[:, 0], transformed_cloud2[:, 1], 
                       c='blue', alpha=0.6, label='Cloud 2 (Optimized)')
            ax3.set_title('Top View (XY Plane)')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.legend()
            ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

# 사용 예제
def generate_sample_clouds():
    """샘플 포인트 클라우드 생성"""
    np.random.seed(42)
    
    # Cloud 1: 구 형태
    n1 = 100
    phi = np.random.uniform(0, 2*np.pi, n1)
    theta = np.random.uniform(0, np.pi, n1)
    r = np.random.uniform(0.8, 1.2, n1)
    
    cloud1 = np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])
    
    # Cloud 2: 원통 형태
    n2 = 80
    theta = np.random.uniform(0, 2*np.pi, n2)
    h = np.random.uniform(-1, 1, n2)
    r = np.random.uniform(0.5, 0.8, n2)
    
    cloud2 = np.column_stack([
        r * np.cos(theta) + 3,  # x축으로 떨어뜨려 놓음
        r * np.sin(theta),
        h
    ])
    
    return cloud1, cloud2

def main():
    """메인 실행 함수"""
    print("포인트 클라우드 최적화 시작...")
    
    # 샘플 데이터 생성
    cloud1, cloud2 = generate_sample_clouds()
    
    # 최적화 객체 생성
    optimizer = PointCloudOptimizer(cloud1, cloud2, min_distance=0.2)
    
    print("초기 상태 시각화...")
    optimizer.visualize()
    
    print("최적화 실행 중...")
    result = optimizer.optimize()
    
    print(f"최적화 완료!")
    print(f"성공 여부: {result.success}")
    print(f"최종 목적함수 값: {result.fun:.6f}")
    print(f"변환 파라미터:")
    print(f"  Translation: {result.x[:3]}")
    print(f"  Rotation (rad): {result.x[3:6]}")
    
    # 최종 거리 계산
    translation = result.x[:3]
    rotation = result.x[3:6]
    transformed_cloud2 = optimizer.transform_cloud(cloud2, translation, rotation)
    final_distance = optimizer.compute_closest_distance(cloud1, transformed_cloud2)
    print(f"  최종 최소 거리: {final_distance:.4f}")
    
    print("최적화 결과 시각화...")
    optimizer.visualize(result)

if __name__ == "__main__":
    main()