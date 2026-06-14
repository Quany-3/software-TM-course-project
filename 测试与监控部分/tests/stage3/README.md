# Stage 3: Selenium and JMeter Testing

This folder contains baseline test assets for Online Boutique.

## 1. Expose the frontend

Use a stable local URL for both Selenium and JMeter:

```powershell
kubectl port-forward svc/frontend-external 8080:80
```

Base URL:

```text
http://localhost:8080
```

## 2. Selenium IDE functional test

Open Selenium IDE, then import:

```text
tests/stage3/online-boutique-functional.side
```

Run the test named:

```text
Browse product and place order
```

The test covers:

- Home page loading
- Product detail page loading
- Add to cart
- Cart page rendering
- Checkout form submission
- Order confirmation page

This project has no login feature, so login testing is not applicable.

## 3. JMeter performance test

Open Apache JMeter, then open:

```text
tests/stage3/online-boutique-performance.jmx
```

Default variables:

```text
host = localhost
port = 8080
threads = 10
ramp_up = 10
duration = 120
```

Run the test from JMeter GUI for debugging. For a formal run, use CLI:

```powershell
jmeter -n -t tests\stage3\online-boutique-performance.jmx -l tests\stage3\results.jtl -e -o tests\stage3\report
```

Metrics to record:

- Average response time
- 90th/95th percentile response time
- Throughput
- Error rate
- CPU and memory from Grafana/Prometheus

## 4. Useful Prometheus queries

```promql
sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default", container!="", image!=""}[5m]))
```

```promql
sum by (pod) (container_memory_working_set_bytes{namespace="default", container!="", image!=""})
```

```promql
sum(rate(container_network_receive_bytes_total{namespace="default"}[5m]))
```

