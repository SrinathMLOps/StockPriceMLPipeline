#!/usr/bin/env python3
# AWS Cloud Deployment Script

import boto3
import json
import os
from datetime import datetime

class AWSDeployer:
    def __init__(self):
        self.region = 'us-east-1'
        self.app_name = 'stock-price-ml-pipeline'
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    def create_ecr_repository(self):
        """Create ECR repository for Docker images"""
        print("🐳 Creating ECR repository...")
        
        ecr_client = boto3.client('ecr', region_name=self.region)
        
        try:
            response = ecr_client.create_repository(
                repositoryName=f'{self.app_name}-fastapi',
                imageScanningConfiguration={'scanOnPush': True}
            )
            print(f"✅ ECR repository created: {response['repository']['repositoryUri']}")
            return response['repository']['repositoryUri']
        except ecr_client.exceptions.RepositoryAlreadyExistsException:
            print("✅ ECR repository already exists")
            response = ecr_client.describe_repositories(repositoryNames=[f'{self.app_name}-fastapi'])
            return response['repositories'][0]['repositoryUri']
    
    def create_ecs_cluster(self):
        """Create ECS cluster for container orchestration"""
        print("🚀 Creating ECS cluster...")
        
        ecs_client = boto3.client('ecs', region_name=self.region)
        
        try:
            response = ecs_client.create_cluster(
                clusterName=f'{self.app_name}-cluster',
                capacityProviders=['FARGATE'],
                defaultCapacityProviderStrategy=[
                    {
                        'capacityProvider': 'FARGATE',
                        'weight': 1
                    }
                ]
            )
            print(f"✅ ECS cluster created: {response['cluster']['clusterName']}")
            return response['cluster']['clusterArn']
        except Exception as e:
            print(f"⚠️ Cluster might already exist: {e}")
            return f"arn:aws:ecs:{self.region}:123456789012:cluster/{self.app_name}-cluster"
    
    def create_rds_instance(self):
        """Create RDS PostgreSQL instance"""
        print("🗄️ Creating RDS PostgreSQL instance...")
        
        rds_client = boto3.client('rds', region_name=self.region)
        
        try:
            response = rds_client.create_db_instance(
                DBInstanceIdentifier=f'{self.app_name}-postgres',
                DBInstanceClass='db.t3.micro',
                Engine='postgres',
                MasterUsername='airflow',
                MasterUserPassword='airflow123',
                AllocatedStorage=20,
                VpcSecurityGroupIds=['sg-default'],
                PubliclyAccessible=True,
                BackupRetentionPeriod=7,
                MultiAZ=False
            )
            print(f"✅ RDS instance creating: {response['DBInstance']['DBInstanceIdentifier']}")
            return response['DBInstance']['Endpoint']['Address']
        except Exception as e:
            print(f"⚠️ RDS creation error: {e}")
            return f"{self.app_name}-postgres.amazonaws.com"
    
    def create_elasticache_cluster(self):
        """Create ElastiCache Redis cluster"""
        print("⚡ Creating ElastiCache Redis cluster...")
        
        elasticache_client = boto3.client('elasticache', region_name=self.region)
        
        try:
            response = elasticache_client.create_cache_cluster(
                CacheClusterId=f'{self.app_name}-redis',
                CacheNodeType='cache.t3.micro',
                Engine='redis',
                NumCacheNodes=1,
                Port=6379
            )
            print(f"✅ ElastiCache cluster creating: {response['CacheCluster']['CacheClusterId']}")
            return f"{self.app_name}-redis.cache.amazonaws.com"
        except Exception as e:
            print(f"⚠️ ElastiCache creation error: {e}")
            return f"{self.app_name}-redis.cache.amazonaws.com"
    
    def create_task_definition(self, ecr_uri, rds_endpoint, redis_endpoint):
        """Create ECS task definition"""
        print("📋 Creating ECS task definition...")
        
        task_definition = {
            "family": f"{self.app_name}-task",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "256",
            "memory": "512",
            "executionRoleArn": f"arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
            "containerDefinitions": [
                {
                    "name": "fastapi",
                    "image": ecr_uri,
                    "portMappings": [
                        {
                            "containerPort": 8000,
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {
                            "name": "POSTGRES_HOST",
                            "value": rds_endpoint
                        },
                        {
                            "name": "REDIS_HOST", 
                            "value": redis_endpoint
                        },
                        {
                            "name": "MLFLOW_TRACKING_URI",
                            "value": "http://mlflow:5000"
                        }
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{self.app_name}",
                            "awslogs-region": self.region,
                            "awslogs-stream-prefix": "ecs"
                        }
                    }
                }
            ]
        }
        
        with open('ecs-task-definition.json', 'w') as f:
            json.dump(task_definition, f, indent=2)
        
        print("✅ Task definition saved to ecs-task-definition.json")
        return task_definition
    
    def create_application_load_balancer(self):
        """Create Application Load Balancer"""
        print("⚖️ Creating Application Load Balancer...")
        
        elbv2_client = boto3.client('elbv2', region_name=self.region)
        
        try:
            response = elbv2_client.create_load_balancer(
                Name=f'{self.app_name}-alb',
                Subnets=['subnet-12345', 'subnet-67890'],  # Replace with actual subnet IDs
                SecurityGroups=['sg-default'],
                Scheme='internet-facing',
                Type='application',
                IpAddressType='ipv4'
            )
            print(f"✅ ALB created: {response['LoadBalancers'][0]['DNSName']}")
            return response['LoadBalancers'][0]['LoadBalancerArn']
        except Exception as e:
            print(f"⚠️ ALB creation error: {e}")
            return f"arn:aws:elasticloadbalancing:{self.region}:123456789012:loadbalancer/app/{self.app_name}-alb"
    
    def create_deployment_scripts(self):
        """Create deployment automation scripts"""
        print("📜 Creating deployment scripts...")
        
        # Docker build and push script
        docker_script = f"""#!/bin/bash
# Docker build and push to ECR

# Get ECR login token
aws ecr get-login-password --region {self.region} | docker login --username AWS --password-stdin {self.app_name}-fastapi.dkr.ecr.{self.region}.amazonaws.com

# Build and tag image
docker build -t {self.app_name}-fastapi ./serving
docker tag {self.app_name}-fastapi:latest {self.app_name}-fastapi.dkr.ecr.{self.region}.amazonaws.com/{self.app_name}-fastapi:latest

# Push to ECR
docker push {self.app_name}-fastapi.dkr.ecr.{self.region}.amazonaws.com/{self.app_name}-fastapi:latest

echo "✅ Docker image pushed to ECR"
"""
        
        with open('deploy_docker.sh', 'w') as f:
            f.write(docker_script)
        
        # ECS deployment script
        ecs_script = f"""#!/bin/bash
# Deploy to ECS

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create or update service
aws ecs create-service \\
    --cluster {self.app_name}-cluster \\
    --service-name {self.app_name}-service \\
    --task-definition {self.app_name}-task \\
    --desired-count 2 \\
    --launch-type FARGATE \\
    --network-configuration "awsvpcConfiguration={{subnets=[subnet-12345,subnet-67890],securityGroups=[sg-default],assignPublicIp=ENABLED}}"

echo "✅ ECS service deployed"
"""
        
        with open('deploy_ecs.sh', 'w') as f:
            f.write(ecs_script)
        
        # Make scripts executable
        os.chmod('deploy_docker.sh', 0o755)
        os.chmod('deploy_ecs.sh', 0o755)
        
        print("✅ Deployment scripts created")
    
    def create_terraform_config(self):
        """Create Terraform infrastructure as code"""
        print("🏗️ Creating Terraform configuration...")
        
        terraform_config = f"""
# AWS Provider
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "{self.region}"
}}

# ECR Repository
resource "aws_ecr_repository" "fastapi" {{
  name                 = "{self.app_name}-fastapi"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {{
    scan_on_push = true
  }}
}}

# ECS Cluster
resource "aws_ecs_cluster" "main" {{
  name = "{self.app_name}-cluster"

  capacity_providers = ["FARGATE"]

  default_capacity_provider_strategy {{
    capacity_provider = "FARGATE"
    weight           = 100
  }}
}}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {{
  identifier     = "{self.app_name}-postgres"
  engine         = "postgres"
  engine_version = "13.7"
  instance_class = "db.t3.micro"
  
  allocated_storage = 20
  storage_encrypted = true
  
  db_name  = "airflow"
  username = "airflow"
  password = "airflow123"
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
}}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {{
  cluster_id           = "{self.app_name}-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  security_group_ids   = [aws_security_group.redis.id]
}}

# Application Load Balancer
resource "aws_lb" "main" {{
  name               = "{self.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids

  enable_deletion_protection = false
}}

# Security Groups
resource "aws_security_group" "alb" {{
  name_prefix = "{self.app_name}-alb"
  
  ingress {{
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
}}

resource "aws_security_group" "rds" {{
  name_prefix = "{self.app_name}-rds"
  
  ingress {{
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }}
}}

resource "aws_security_group" "redis" {{
  name_prefix = "{self.app_name}-redis"
  
  ingress {{
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }}
}}

# Data sources
data "aws_vpc" "default" {{
  default = true
}}

data "aws_subnets" "default" {{
  filter {{
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }}
}}

# Outputs
output "ecr_repository_url" {{
  value = aws_ecr_repository.fastapi.repository_url
}}

output "load_balancer_dns" {{
  value = aws_lb.main.dns_name
}}

output "rds_endpoint" {{
  value = aws_db_instance.postgres.endpoint
}}

output "redis_endpoint" {{
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}}
"""
        
        with open('main.tf', 'w') as f:
            f.write(terraform_config)
        
        print("✅ Terraform configuration created: main.tf")
    
    def deploy_to_aws(self):
        """Complete AWS deployment process"""
        print("🚀 Starting AWS Deployment Process")
        print("=" * 50)
        
        # Create infrastructure
        ecr_uri = self.create_ecr_repository()
        cluster_arn = self.create_ecs_cluster()
        rds_endpoint = self.create_rds_instance()
        redis_endpoint = self.create_elasticache_cluster()
        
        # Create configurations
        self.create_task_definition(ecr_uri, rds_endpoint, redis_endpoint)
        alb_arn = self.create_application_load_balancer()
        
        # Create deployment scripts
        self.create_deployment_scripts()
        self.create_terraform_config()
        
        print("\n✅ AWS deployment configuration complete!")
        print("\nNext steps:")
        print("1. Configure AWS CLI: aws configure")
        print("2. Deploy infrastructure: terraform init && terraform apply")
        print("3. Build and push Docker: ./deploy_docker.sh")
        print("4. Deploy to ECS: ./deploy_ecs.sh")
        print("5. Access via ALB DNS name")
        
        return True

if __name__ == "__main__":
    deployer = AWSDeployer()
    deployer.deploy_to_aws()