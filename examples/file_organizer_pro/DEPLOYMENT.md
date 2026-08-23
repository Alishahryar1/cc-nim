# File Organizer Pro - Deployment Guide

This guide covers deploying File Organizer Pro to production environments using Docker and Kubernetes.

## Prerequisites

### Docker Deployment
- Docker installed and running
- Docker Compose (optional, but recommended)
- 2GB+ free disk space

### Kubernetes Deployment
- Kubernetes cluster (1.24+)
- kubectl configured and authenticated
- 3+ worker nodes recommended for high availability

---

## Quick Start: Docker

### 1. Build the Image

```bash
cd examples/file_organizer_pro
docker build -t file-organizer-pro:latest .
```

### 2. Run with Docker Compose

```bash
docker-compose up -d
```

This starts the API server accessible at `http://localhost:8000`.

### 3. Verify Deployment

```bash
# Check if container is running
docker ps | grep file-organizer

# View logs
docker-compose logs -f file-organizer-api

# Test the API
curl http://localhost:8000/
```

### 4. Stop the Service

```bash
docker-compose down
```

---

## Docker Customization

### Using Environment Variables

```bash
# Custom logging level
docker run -e LOG_LEVEL=DEBUG -p 8000:8000 file-organizer-pro:latest

# Mount custom data directory
docker run -v /custom/path:/data -p 8000:8000 file-organizer-pro:latest
```

### Scaling with Docker Compose

To run multiple instances behind a load balancer:

```yaml
version: '3.8'
services:
  api-1:
    build: .
    ports:
      - "8001:8000"
  api-2:
    build: .
    ports:
      - "8002:8000"
  api-3:
    build: .
    ports:
      - "8003:8000"
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api-1
      - api-2
      - api-3
```

---

## Kubernetes Deployment

### 1. Create the Namespace and Resources

```bash
kubectl apply -f kubernetes.yaml
```

This creates:
- Namespace: `file-organizer`
- Deployment with 3 replicas
- Service (ClusterIP)
- HorizontalPodAutoscaler (3-10 replicas)
- PodDisruptionBudget (minimum 2 available)

### 2. Verify Deployment

```bash
# Check pods
kubectl get pods -n file-organizer

# View deployment status
kubectl describe deployment file-organizer-pro -n file-organizer

# Check service
kubectl get svc -n file-organizer
```

### 3. Access the API

#### Port Forward (Development)

```bash
kubectl port-forward -n file-organizer svc/file-organizer-service 8000:80
# Access at http://localhost:8000
```

#### Ingress (Production)

Create an ingress resource:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: file-organizer-ingress
  namespace: file-organizer
spec:
  ingressClassName: nginx
  rules:
  - host: file-organizer.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: file-organizer-service
            port:
              number: 80
  tls:
  - hosts:
    - file-organizer.example.com
    secretName: file-organizer-tls
```

```bash
kubectl apply -f ingress.yaml
```

### 4. Monitor Deployment

```bash
# Watch pod creation
kubectl get pods -n file-organizer --watch

# View logs
kubectl logs -n file-organizer -l app=file-organizer -f

# Check autoscaler status
kubectl get hpa -n file-organizer
```

### 5. Scale Manually (Optional)

```bash
# Scale to specific number of replicas
kubectl scale deployment file-organizer-pro -n file-organizer --replicas=5

# HPA will manage scaling automatically between minReplicas (3) and maxReplicas (10)
```

---

## Production Checklist

- [ ] Image built and tagged with version (e.g., `file-organizer-pro:v1.0.0`)
- [ ] Environment variables configured (LOG_LEVEL, resource limits)
- [ ] Persistent volume configured for data (if needed)
- [ ] Health checks passing (liveness and readiness probes)
- [ ] Resource requests and limits set appropriately
- [ ] Secrets configured (if API key authentication added)
- [ ] Monitoring and logging configured (Prometheus, ELK, CloudWatch)
- [ ] Backups configured for task history
- [ ] Load balancing and ingress controller deployed
- [ ] SSL/TLS certificates installed (for Kubernetes)
- [ ] Network policies configured for security
- [ ] Testing completed (smoke tests, load tests)

---

## Troubleshooting

### Docker Issues

**Container exits immediately:**
```bash
# Check logs for error
docker logs <container_id>

# Run with interactive terminal
docker run -it file-organizer-pro:latest
```

**Port already in use:**
```bash
# Find what's using port 8000
lsof -i :8000

# Use different port
docker run -p 8001:8000 file-organizer-pro:latest
```

### Kubernetes Issues

**Pods not starting:**
```bash
# Check pod status and events
kubectl describe pod <pod_name> -n file-organizer

# View pod logs
kubectl logs <pod_name> -n file-organizer
```

**Service not accessible:**
```bash
# Check endpoints
kubectl get endpoints -n file-organizer

# Test service DNS
kubectl run -it --rm debug --image=alpine:latest --restart=Never -- sh
# Inside pod: nslookup file-organizer-service.file-organizer
```

**Memory or CPU issues:**
```bash
# View resource usage
kubectl top pods -n file-organizer

# Adjust requests/limits in deployment spec
kubectl edit deployment file-organizer-pro -n file-organizer
```

---

## Performance Tuning

### Docker

```bash
# Increase memory limit
docker run -m 1g file-organizer-pro:latest

# Set CPU limits
docker run --cpus="2" file-organizer-pro:latest

# Use volume mount for better I/O
docker run -v /fast/ssd:/data file-organizer-pro:latest
```

### Kubernetes

Adjust in `kubernetes.yaml`:

```yaml
resources:
  requests:
    memory: "512Mi"    # Increase for large tasks
    cpu: "500m"        # Increase for parallelization
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

Update HPA thresholds:

```yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 80  # Scale at 80% CPU usage
```

---

## Security Considerations

1. **Image Security**
   - Use specific Python version tag (not `latest`)
   - Scan images for vulnerabilities: `docker scan file-organizer-pro`
   - Use private registry for production

2. **Network Security**
   - Restrict API access with authentication
   - Use TLS/SSL for all connections
   - Implement network policies in Kubernetes

3. **Data Security**
   - Encrypt sensitive data at rest
   - Use secure volume mounts
   - Implement access controls for organized files

4. **Resource Limits**
   - Set memory and CPU limits (prevents resource exhaustion)
   - Configure request timeouts
   - Rate limit API endpoints

---

## Monitoring & Logging

### Prometheus Metrics

The deployment already exposes metrics at `/metrics`. Configure Prometheus:

```yaml
scrape_configs:
  - job_name: 'file-organizer'
    static_configs:
      - targets: ['file-organizer-service.file-organizer:8000']
```

### Centralized Logging

Send logs to ELK stack or cloud services:

```bash
# Docker: Use logging driver
docker run --log-driver splunk \
  --log-opt splunk-token=<TOKEN> \
  --log-opt splunk-url=https://<SPLUNK_HOST>:8088 \
  file-organizer-pro:latest
```

### Alert Rules

Configure alerts for:
- Pod restart frequency (> 3 in 10 min)
- High memory usage (> 80%)
- High CPU usage (> 80%)
- API error rates (> 5%)

---

## Updating Deployment

### Docker

```bash
# Rebuild image
docker build -t file-organizer-pro:v1.1.0 .

# Update service
docker-compose down
docker-compose up -d  # Uses updated image
```

### Kubernetes

```bash
# Update image in deployment
kubectl set image deployment/file-organizer-pro \
  -n file-organizer \
  api=file-organizer-pro:v1.1.0

# Verify rollout
kubectl rollout status deployment/file-organizer-pro -n file-organizer

# Rollback if needed
kubectl rollout undo deployment/file-organizer-pro -n file-organizer
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy File Organizer Pro

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t file-organizer-pro:${{ github.sha }} .
      
      - name: Push to registry
        run: docker push file-organizer-pro:${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/file-organizer-pro \
            -n file-organizer \
            api=file-organizer-pro:${{ github.sha }}
```

---

## Support & Debugging

For detailed logs and debugging:

```bash
# Docker
export LOG_LEVEL=DEBUG
docker-compose up

# Kubernetes
kubectl set env deployment/file-organizer-pro \
  -n file-organizer \
  LOG_LEVEL=DEBUG
```

For additional help:
- Review main [FILE_ORGANIZER_PRO_GUIDE.md](FILE_ORGANIZER_PRO_GUIDE.md)
- Check FastAPI docs: https://fastapi.tiangolo.com
- View Kubernetes docs: https://kubernetes.io/docs/
