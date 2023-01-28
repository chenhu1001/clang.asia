---
title: 基于令牌桶算法的Java限流实现
date: 2023-01-28 21:24:43
categories: Java
tags: [Java,算法,秒杀]
---
令牌桶算法是一种流量限制算法，通常用于限制对服务的请求速率。它通过维护一个桶来存储令牌来实现限流。每当有请求需要被处理时，都需要先从桶中获取一个令牌。如果桶中有可用令牌，请求就会被处理，并且令牌会被从桶中移除。如果桶中没有可用令牌，请求就会被拒绝或等待。

桶中的令牌是按固定的速率进行填充的，这个速率就是限流的速率。这种算法的优点是它可以平滑地处理请求，可以避免突发请求造成的服务崩溃。

令牌桶算法在多种场景下都有应用，如在网络流量控制、服务限流、网络防火墙等场景都可以应用这种算法。

首先，需要创建一个桶来存储令牌，这里可以使用 Java 的 Semaphore 类来实现。Semaphore 类是一个信号量类，可以用来控制线程的并发访问。

接下来，需要启动一个线程来不断地向桶中填充令牌。这里可以使用 Java 的 ScheduledExecutorService 来实现。ScheduledExecutorService 可以让你在给定的延迟之后或者在给定的间隔之后执行任务。

当请求需要被处理时，程序会从桶中尝试获取一个令牌。如果桶中有可用令牌，请求就会被处理，并且令牌会被从桶中移除。如果桶中没有可用令牌，请求就会被拒绝或等待。

在这个过程中，Semaphore 类的 acquire() 方法和 release() 方法用来获取和释放令牌， ScheduledExecutorService 的 scheduleAtFixedRate() 方法用来在固定的时间间隔内执行任务。

这个算法的优点是它可以平滑地处理请求，可以避免突发请求造成的服务崩溃。

以下是一个基于令牌桶算法的 Java 限流代码示例：
```
import java.util.concurrent.Semaphore;

public class RateLimiter {
    private final Semaphore semaphore;
    private final int maxPermits;
    private final long period;
    private final TimeUnit timeUnit;
    private final ScheduledExecutorService scheduler;

    public RateLimiter(int permits, long period, TimeUnit timeUnit) {
        this.semaphore = new Semaphore(permits);
        this.maxPermits = permits;
        this.period = period;
        this.timeUnit = timeUnit;
        this.scheduler = Executors.newScheduledThreadPool(1);
        scheduler.scheduleAtFixedRate(new Replenisher(), period, period, timeUnit);
    }

    public boolean acquire() {
        return semaphore.tryAcquire();
    }

    public void release() {
        semaphore.release();
    }

    private class Replenisher implements Runnable {
        public void run() {
            semaphore.release(maxPermits - semaphore.availablePermits());
        }
    }
}
```
这个类实现了一个令牌桶限流器，可以通过调用 acquire() 方法来获取一个令牌，如果桶中没有可用的令牌，则 acquire() 方法会返回 false。 每次调用 release() 方法都会增加一个令牌。
